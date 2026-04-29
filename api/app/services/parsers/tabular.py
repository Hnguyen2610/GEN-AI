import os
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime, timezone
import pandas as pd
import duckdb
from typing import Any

from api.app.models.entities import DatasetSheet, ColumnProfile

class TabularParser:
    def __init__(self, storage_root: str):
        self.storage_root = Path(storage_root)
        self.duckdb_dir = self.storage_root / "duckdb"
        self.duckdb_dir.mkdir(parents=True, exist_ok=True)

    def _get_duckdb_path(self, workspace_id: UUID, dataset_id: UUID) -> str:
        workspace_dir = self.duckdb_dir / str(workspace_id)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return str(workspace_dir / f"{dataset_id}.duckdb")

    def parse_and_materialize(
        self, 
        file_path: str, 
        workspace_id: UUID, 
        dataset_id: UUID, 
        dataset_version_id: UUID
    ) -> tuple[list[DatasetSheet], list[ColumnProfile]]:
        extension = os.path.splitext(file_path)[1].lower()
        db_path = self._get_duckdb_path(workspace_id, dataset_id)
        
        sheets_meta: list[DatasetSheet] = []
        profiles: list[ColumnProfile] = []

        # Connect to DuckDB
        conn = duckdb.connect(db_path)
        try:
            if extension == ".csv":
                df = pd.read_csv(file_path)
                sheet_name = "default"
                self._process_sheet(
                    conn, df, sheet_name, workspace_id, dataset_id, 
                    dataset_version_id, sheets_meta, profiles
                )
            elif extension in [".xlsx", ".xls"]:
                excel = pd.ExcelFile(file_path)
                for sheet_name in excel.sheet_names:
                    df = excel.parse(sheet_name)
                    self._process_sheet(
                        conn, df, sheet_name, workspace_id, dataset_id, 
                        dataset_version_id, sheets_meta, profiles
                    )
        finally:
            conn.close()

        return sheets_meta, profiles

    def _process_sheet(
        self,
        conn: duckdb.DuckDBPyConnection,
        df: pd.DataFrame,
        sheet_name: str,
        workspace_id: UUID,
        dataset_id: UUID,
        dataset_version_id: UUID,
        sheets_meta: list[DatasetSheet],
        profiles: list[ColumnProfile]
    ):
        # Normalize sheet name for DuckDB table name and quote it
        table_name = f'"{sheet_name.replace(" ", "_").replace("-", "_").lower()}"'
        
        # Ingest into DuckDB
        conn.register("tmp_df", df)
        conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM tmp_df")
        conn.unregister("tmp_df")

        # Metadata: Sheet
        sheet = DatasetSheet(
            id=uuid4(),
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            asset_version_id=dataset_version_id,
            name=sheet_name,
            row_count=len(df),
            column_count=len(df.columns)
        )
        sheets_meta.append(sheet)

        # Metadata: Column Profiles
        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            
            # Basic stats (safely calculate min/max even with mixed types)
            null_count = int(series.isnull().sum())
            distinct_count = int(series.nunique())
            
            try:
                min_val = str(series.min()) if not series.isnull().all() and hasattr(series, 'min') else None
            except TypeError:
                # Handle mixed types (e.g. float and str) by converting to string first for lexicographical min
                try:
                    min_val = str(series.dropna().astype(str).min())
                except Exception:
                    min_val = None

            try:
                max_val = str(series.max()) if not series.isnull().all() and hasattr(series, 'max') else None
            except TypeError:
                try:
                    max_val = str(series.dropna().astype(str).max())
                except Exception:
                    max_val = None

            
            # Sample values (top 5 distinct non-null)
            samples = series.dropna().unique()[:5].tolist()
            sample_dict = {"values": [str(s) for s in samples]}

            profile = ColumnProfile(
                id=uuid4(),
                dataset_id=dataset_id,
                dataset_version_id=dataset_version_id,
                asset_version_id=dataset_version_id,
                sheet_name=sheet_name,
                column_name=str(col),
                data_type=dtype,
                null_count=null_count,
                distinct_count=distinct_count,
                min_value=min_val,
                max_value=max_val,
                sample_values=sample_dict
            )
            profiles.append(profile)

    def get_preview(self, workspace_id: UUID, dataset_id: UUID, sheet_name: str | None = None, limit: int = 20) -> dict[str, Any]:
        db_path = self._get_duckdb_path(workspace_id, dataset_id)
        if not os.path.exists(db_path):
             return {"columns": [], "rows": []}
             
        conn = duckdb.connect(db_path, read_only=True)
        try:
            if not sheet_name:
                tables = conn.execute("SHOW TABLES").fetchall()
                if not tables:
                    return {"columns": [], "rows": []}
                table_name = f'"{tables[0][0]}"'
            else:
                table_name = f'"{sheet_name.replace(" ", "_").replace("-", "_").lower()}"'

            res = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").df()
            return {
                "sheet_name": sheet_name or "default",
                "columns": res.columns.tolist(),
                "rows": res.to_dict(orient="records")
            }
        finally:
            conn.close()

    def delete_materialization(self, workspace_id: UUID, dataset_id: UUID) -> None:
        db_path = self._get_duckdb_path(workspace_id, dataset_id)
        for ext in ["", ".wal", ".tmp"]:
            full_path = db_path + ext
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except Exception:
                    pass
