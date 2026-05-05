import os
from uuid import UUID, uuid4
from typing import Any
import docx
# from pypdf import PdfReader # Moved to lazy import
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
import httpx

from api.app.models.entities import KnowledgeChunk

class KnowledgeParser:
    def __init__(
        self,
        gemini_api_key: str | None = None,
        ollama_url: str | None = None,
        ollama_embed_model: str = "nomic-embed-text",
        gemini_embed_model: str = "gemini-embedding-001",
    ):
        self.gemini_api_key = gemini_api_key
        self.ollama_url = ollama_url
        self.ollama_embed_model = ollama_embed_model
        self.gemini_embed_model = gemini_embed_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        self.client = None
        if self.gemini_api_key:
            self.client = genai.Client(api_key=self.gemini_api_key)

    def _extract_docx_text(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        blocks: list[str] = []

        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if text:
                    blocks.append(text)
                
                if self.client:
                    drawings = child.xpath('.//w:drawing')
                    for drawing in drawings:
                        blips = drawing.xpath('.//a:blip')
                        for blip in blips:
                            rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                            if rId and rId in doc.part.related_parts:
                                part = doc.part.related_parts[rId]
                                try:
                                    response = self.client.models.generate_content(
                                        model='gemini-2.5-flash',
                                        contents=[
                                            types.Part.from_bytes(data=part.blob, mime_type=part.content_type),
                                            "Bạn là một chuyên gia nhận diện văn bản OCR. Trích xuất TẤT CẢ chữ và bảng từ hình ảnh này sang định dạng text/markdown. Chỉ trả lời kết quả, không bình luận thêm."
                                        ]
                                    )
                                    if response.text:
                                        text_result = response.text.strip()
                                        blocks.append(text_result)
                                        import logging as _log
                                        _log.getLogger(__name__).info(
                                            "OCR success for image rId=%s (%d chars)", rId, len(text_result)
                                        )
                                except Exception as e:
                                    import logging as _log
                                    _log.getLogger(__name__).warning(
                                        "OCR via Gemini Vision failed for image rId=%s: %s. "
                                        "Text-only chunks will still be saved.",
                                        rId, e
                                    )
                                    # Do NOT re-raise: OCR is best-effort.
                                    # Text paragraphs/tables are still valid and will be indexed.
            elif isinstance(child, CT_Tbl):
                table = Table(child, doc)
                row_texts: list[str] = []
                for row in table.rows:
                    cells = [cell.text.replace("\n", " ").strip() if cell.text and cell.text.strip() else "-" for cell in row.cells]
                    row_texts.append(" | ".join(cells))
                if row_texts:
                    blocks.append("\n".join(row_texts))

        return "\n\n".join(blocks)

    def _extract_pdf_text_with_vision(self, file_path: str) -> str:
        if not self.client:
            return ""
        
        try:
            import logging as _log
            _log.getLogger(__name__).info("Running Vision-based extraction for PDF: %s", file_path)
            
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
                    "Bạn là một chuyên gia phân tích tài liệu PDF. Trích xuất toàn bộ nội dung văn bản và bảng biểu từ file PDF này sang định dạng Markdown. Giữ nguyên cấu trúc bảng. Chỉ trả về kết quả, không bình luận thêm."
                ]
            )
            return response.text.strip() if response.text else ""
        except Exception as e:
            import logging as _log
            _log.getLogger(__name__).warning("PDF Vision extraction failed: %s. Falling back to text-based extraction.", e)
            return ""

    def parse_and_chunk(self, file_path: str) -> list[dict[str, Any]]:
        extension = os.path.splitext(file_path)[1].lower()
        full_text = ""

        if extension == ".pdf":
            # Try Vision-based extraction if Gemini is available for better quality/OCR
            if self.client:
                full_text = self._extract_pdf_text_with_vision(file_path)
            
            # Fallback to standard pypdf if Vision failed or no client
            if not full_text:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    full_text += (page.extract_text() or "") + "\n"
        elif extension == ".docx":
            full_text = self._extract_docx_text(file_path)
        elif extension in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
        
        if not full_text.strip():
            return []

        chunks = self.splitter.split_text(full_text)
        return [{"content": chunk, "index": i} for i, chunk in enumerate(chunks)]

    async def generate_embeddings(self, chunks: list[dict[str, Any]]) -> list[list[float]]:
        texts = [c["content"] for c in chunks]
        embeddings = []
        
        # 1. Use Ollama local model if URL is provided (Priority to save Quota)
        if self.ollama_url:
            async with httpx.AsyncClient() as client:
                for text in texts:
                    response = await client.post(
                        f"{self.ollama_url.rstrip('/')}/api/embeddings",
                        json={"model": self.ollama_embed_model, "prompt": text},
                        timeout=30.0
                    )
                    response.raise_for_status()
                    data = response.json()
                    embeddings.append(data["embedding"])
            return embeddings
            
        # 2. Fallback to Gemini Embeddings
        if self.client:
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                result = self.client.models.embed_content(
                    model=self.gemini_embed_model,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )
                embeddings.extend([e.values for e in result.embeddings])
            return embeddings

        # 3. Fallback dummy
        return [[0.1] * 768 for _ in chunks]

    async def process_file(
        self, 
        file_path: str, 
        workspace_id: UUID, 
        knowledge_asset_id: UUID, 
        knowledge_version_id: UUID
    ) -> list[KnowledgeChunk]:
        chunk_data = self.parse_and_chunk(file_path)
        if not chunk_data:
            raise ValueError("Knowledge asset produced no extractable text chunks")
             
        embeddings = await self.generate_embeddings(chunk_data)
        
        knowledge_chunks = []
        for i, (data, embedding) in enumerate(zip(chunk_data, embeddings)):
            chunk = KnowledgeChunk(
                id=uuid4(),
                knowledge_version_id=knowledge_version_id,
                asset_version_id=knowledge_version_id,
                chunk_index=data["index"],
                content=data["content"],
                embedding=embedding,
            )
            knowledge_chunks.append(chunk)
            
        return knowledge_chunks
