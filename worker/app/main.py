import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from api.app.models.entities import AssetKind, JobStatus
from api.app.services.parsers.knowledge import KnowledgeParser
from api.app.services.parsers.tabular import TabularParser
from worker.app.core.settings import get_settings
from worker.app.services.processors import KnowledgeLaneProcessor, StructuredLaneProcessor
from worker.app.services.storage import StorageReader

logger = logging.getLogger(__name__)


async def claim_next_job(session: AsyncSession):
    claim_sql = text("""
        UPDATE ingestion_jobs
        SET status = :processing_status, updated_at = NOW()
        WHERE id = (
            SELECT id FROM ingestion_jobs
            WHERE status = :pending_status
              AND asset_id IS NOT NULL
              AND asset_version_id IS NOT NULL
              AND asset_kind IS NOT NULL
            ORDER BY created_at ASC
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, asset_id, asset_version_id, asset_kind, workspace_id;
    """)
    result = await session.execute(
        claim_sql,
        {
            "processing_status": JobStatus.processing.value,
            "pending_status": JobStatus.pending.value,
        },
    )
    row = result.fetchone()
    if row:
        await session.commit()
        return dict(row._mapping)
    return None


async def process_job(
    settings,
    job_info: dict,
    session_maker,
    storage_reader: StorageReader,
    tabular_parser: TabularParser,
    knowledge_parser: KnowledgeParser,
):
    job = None
    processor = None
    try:
        processors = {
            AssetKind.dataset: StructuredLaneProcessor(settings, storage_reader, tabular_parser),
            AssetKind.knowledge: KnowledgeLaneProcessor(settings, storage_reader, knowledge_parser),
        }
        job = _normalize_job(job_info)
        processor = processors[job["asset_kind"]]

        async with session_maker() as session:
            await _mark_asset_status(session, job["asset_id"], JobStatus.processing.value)
            await processor.process(session, job)
            await _mark_asset_status(session, job["asset_id"], JobStatus.ready.value)
            await _mark_job_ready(session, job["id"])
            await session.commit()
            logger.info("Job %s completed successfully", job["id"])
    except Exception as exc:
        logger.exception("Job %s failed", job_info.get("id"))
        asset_id = (job or job_info).get("asset_id")
        async with session_maker() as session:
            if job_info.get("id"):
                await _mark_job_failed(session, job_info["id"], str(exc)[:1000])
            if asset_id:
                await _mark_asset_status(session, asset_id, JobStatus.failed.value)
                if processor is not None:
                    await processor.mark_failed(session, asset_id)
            await session.commit()


def _normalize_job(job_info: dict) -> dict:
    required_fields = ["id", "workspace_id", "asset_id", "asset_version_id", "asset_kind"]
    missing = [field for field in required_fields if not job_info.get(field)]
    if missing:
        raise ValueError(f"Job is missing generic fields: {', '.join(missing)}")

    kind = job_info["asset_kind"]
    if not isinstance(kind, AssetKind):
        kind = AssetKind(kind)

    return {**job_info, "asset_kind": kind}


async def _mark_asset_status(session: AsyncSession, asset_id, status: str) -> None:
    await session.execute(
        text("UPDATE assets SET status = :status, updated_at = NOW() WHERE id = :id"),
        {"id": asset_id, "status": status},
    )


async def _mark_job_ready(session: AsyncSession, job_id) -> None:
    await session.execute(
        text("UPDATE ingestion_jobs SET status = 'ready', updated_at = NOW() WHERE id = :id"),
        {"id": job_id},
    )


async def _mark_job_failed(session: AsyncSession, job_id, error_message: str) -> None:
    await session.execute(
        text(
            "UPDATE ingestion_jobs SET status = 'failed', error_message = :err, "
            "updated_at = NOW() WHERE id = :id"
        ),
        {"id": job_id, "err": error_message},
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    settings = get_settings()

    if settings.worker_mode == "idle":
        logger.info("Worker is in idle mode.")
        while True:
            await asyncio.sleep(60)

    logger.info("Starting production worker...")
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    storage_reader = StorageReader(settings)
    tabular_parser = TabularParser(storage_root=settings.storage_local_path)
    knowledge_parser = KnowledgeParser(
        gemini_api_key=settings.gemini_vision_api_key or settings.gemini_api_key,
        ollama_url=settings.ollama_url,
        ollama_embed_model=settings.ollama_embed_model,
    )

    while True:
        try:
            async with session_maker() as session:
                job_info = await claim_next_job(session)

            if job_info:
                logger.info("Claimed job: %s", job_info["id"])
                await process_job(
                    settings,
                    job_info,
                    session_maker,
                    storage_reader,
                    tabular_parser,
                    knowledge_parser,
                )
                continue
        except Exception:
            logger.exception("Error in worker main loop")

        await asyncio.sleep(settings.poll_interval_seconds)


# Triggering hot-reload to pick up new .env variables
if __name__ == "__main__":
    asyncio.run(main())
