from __future__ import annotations

import os
import time
import uuid
from glob import glob
from typing import List, Dict, Any, Optional

from tenacity import retry, wait_random_exponential, stop_after_attempt

# LangChain / OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_chroma import Chroma # type: ignore
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from pydantic import BaseModel, Field
import tiktoken


class RAGConfig(BaseModel):
    persist_directory: str = Field(..., description="Directory where Chroma collection is stored")
    collection_name: str = Field(default="my_context_db")
    embedding_model: str = Field(default="text-embedding-3-small")
    llm_model: str = Field(default="gpt-4o-mini")
    # The following are only used if you later ingest; kept for compatibility
    chunk_size: int = 2000
    chunk_overlap: int = 200
    neighbor_chars: int = 300
    throttle_seconds: float = 0.25
    summary_max_tokens: int = 600


class RAGQueryRequest(BaseModel):
    question: str
    k: int = 5


class RAGQueryResponse(BaseModel):
    answer: str
    used_k: int


class RAGService:
    def __init__(self, config: RAGConfig):
        self.config = config
        os.makedirs(self.config.persist_directory, exist_ok=True)
        self._embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
        self._chat = ChatOpenAI(
            model_name=self.config.llm_model,
            temperature=0,
            timeout=60,
            max_retries=8,
        )
        # Bound instance for concise summaries
        self._summarizer = self._chat.bind(max_tokens=self.config.summary_max_tokens)
        self._db: Optional[Chroma] = None

    # ---- internal helpers ----
    def _load_db(self) -> Chroma:
        if self._db is None:
            self._db = Chroma(
                persist_directory=self.config.persist_directory,
                collection_name=self.config.collection_name,
                embedding_function=self._embeddings,
            )
        return self._db

    # removed unused _build_ids helper to keep the service minimal

    # ---- LLM calls ----
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(8))
    def _summarize_chunk(self, chunk_text: str, *, title: Optional[str], page: Optional[int]) -> str:
        prefix = f"Title: {title}\nPage: {page}\n" if title else ""
        prompt_tmpl = ChatPromptTemplate.from_template(
            "You are an assistant for road-work document analysis.\n"
            "Summarize the following chunk in 2–3 sentences, suitable for retrieval context.\n"
            "Be specific (places, programs, dates), avoid generalities, no hallucinations.\n\n"
            "{pref}Chunk:\n{chunk}\n\nSummary:"
        )
        chain = prompt_tmpl | self._summarizer | StrOutputParser()
        return chain.invoke({"pref": prefix, "chunk": chunk_text})

    # ---- notebook-equivalent helpers (optional) ----
    @staticmethod
    def clamp_tokens(text: str, max_tokens: int) -> str:
        """Clamp text to a token budget using the o200k_base tokenizer (4o-family)."""
        enc = tiktoken.get_encoding("o200k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        out = enc.decode(toks[:max_tokens])
        cut = max(out.rfind(". "), out.rfind("\n"), out.rfind(" "))
        if cut > 0:
            out = out[: cut + 1]
        return out + " …"

    def create_contextual_chunks(
        self,
        file_path: str,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        neighbor_chars: Optional[int] = None,
        throttle_s: Optional[float] = None,
    ) -> List[Document]:
        """
        Re-implementation of notebook's create_contextual_chunks.
        Returns Documents where page_content = "summary\n\nchunk", and metadata has 'chunk_summary'.
        """
        csize = chunk_size or self.config.chunk_size
        cover = chunk_overlap or self.config.chunk_overlap
        neigh = self.config.neighbor_chars if neighbor_chars is None else neighbor_chars
        sleep_s = self.config.throttle_seconds if throttle_s is None else throttle_s

        loader = PyMuPDFLoader(file_path)
        doc_pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=csize, chunk_overlap=cover)
        doc_chunks = splitter.split_documents(doc_pages)

        contextual: List[Document] = []
        n = len(doc_chunks)
        for i, ch in enumerate(doc_chunks):
            text = ch.page_content
            if neigh and n > 1:
                prev_txt = doc_chunks[i - 1].page_content[-neigh:] if i > 0 else ""
                next_txt = doc_chunks[i + 1].page_content[:neigh] if i + 1 < n else ""
                if prev_txt:
                    text = f"(Prev) {prev_txt}\n\n{text}"
                if next_txt:
                    text = f"{text}\n\n(Next) {next_txt}"

            meta = {
                "id": str(uuid.uuid4()),
                "page": ch.metadata.get("page"),
                "source": ch.metadata.get("source"),
                "title": (ch.metadata.get("source") or "").split("/")[-1],
            }
            summary = self._summarize_chunk(text, title=meta["title"], page=meta["page"])
            contextual.append(
                Document(page_content=f"{summary}\n\n{ch.page_content}", metadata={**meta, "chunk_summary": summary})
            )
            time.sleep(sleep_s)

        return contextual

    def process_files_and_persist(self, pdf_glob: str | List[str]) -> Dict[str, Any]:
        """
        Notebook's processing loop + Chroma.from_documents persist. Optional; not used by API
        when a persisted DB already exists, but provided for completeness.
        """
        file_paths = glob(pdf_glob) if isinstance(pdf_glob, str) else list(pdf_glob)
        paper_docs: List[Document] = []
        for fp in file_paths:
            docs = self.create_contextual_chunks(fp)
            paper_docs.extend(docs)

        if not paper_docs:
            return {"added": 0, "files": 0, "persist_directory": self.config.persist_directory}

        # Build a fresh collection from documents (matches notebook behaviour)
        Chroma.from_documents(
            documents=paper_docs,
            collection_name=self.config.collection_name,
            embedding=self._embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=self.config.persist_directory,
        )
        # Ensure internal handle points to the collection
        self._db = None
        self._load_db()
        return {"added": len(paper_docs), "files": len(file_paths), "persist_directory": self.config.persist_directory}

    # ---- public API ----
    def query(self, question: str, *, k: int = 5) -> Dict[str, Any]:
        db = self._load_db()
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        rag_prompt = (
            "You are an assistant who is an expert in question-answering tasks.\n"
            "Answer the following question using only the following pieces of retrieved context.\n"
            "If the answer is not in the context, say you don't know.\n"
            "Keep the answer detailed and well formatted based on the information from the context.\n\n"
            "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:"
        )
        prompt = ChatPromptTemplate.from_template(rag_prompt)

        chain = ({"context": retriever | format_docs, "question": lambda x: x}) | prompt | self._chat
        res = chain.invoke(question)
        return {"answer": getattr(res, "content", str(res)), "used_k": k}

    def retrieve_context(self, question: str, *, k: int = 5) -> Dict[str, Any]:
        """Return raw retrieved context and metadata without asking the LLM to answer."""
        db = self._load_db()
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)
        meta = [d.metadata for d in docs]
        return {"context": context, "docs": meta, "used_k": k}


# Convenience factory with sensible defaults relative to repo
def make_default_rag_service() -> RAGService:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    persist_dir = os.path.join(repo_root, "notebooks", "my_context_db")
    cfg = RAGConfig(persist_directory=persist_dir)
    return RAGService(cfg)


__all__ = ["RAGService", "RAGConfig", "RAGQueryRequest", "RAGQueryResponse", "make_default_rag_service"]


