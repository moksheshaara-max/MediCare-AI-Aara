"""
MediCare AI - Local MPNet Query Embedder

Runs sentence-transformers/all-mpnet-base-v2 locally on CPU.

IMPORTANT:
- This is the SAME embedding model used by the ingestion pipeline.
- Output is exactly 768 dimensions.
- normalize_embeddings=False matches stored MongoDB embeddings.
- The model is lazy-loaded once and reused for all later requests.
- No Hugging Face paid Inference Provider call is used.
"""

import os
import threading
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer


MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSIONS = 768

# CPU thread count can be overridden later on Render.
TORCH_CPU_THREADS = int(os.getenv("TORCH_CPU_THREADS", "2"))

_model = None
_model_init_lock = threading.Lock()
_encode_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    """
    Lazy-load MPNet once per backend process.

    First query:
        loads model into memory.

    Later queries:
        reuse the same model instance.
    """
    global _model

    if _model is not None:
        return _model

    with _model_init_lock:
        if _model is not None:
            return _model

        torch.set_num_threads(TORCH_CPU_THREADS)

        print(
            f"Loading local query embedder: {MODEL_ID} "
            f"(CPU threads={torch.get_num_threads()})"
        )

        model = SentenceTransformer(
            MODEL_ID,
            device="cpu",
        )

        if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"expected {EMBEDDING_DIMENSIONS}, "
                f"got {model.get_sentence_embedding_dimension()}"
            )

        _model = model

        print(
            f"Local MPNet query embedder ready "
            f"({EMBEDDING_DIMENSIONS} dimensions)"
        )

        return _model


def embed_single_text(
    text: str,
    max_retries: int = 1,
) -> Optional[List[float]]:
    """
    Generate one 768-D embedding using local MPNet.

    max_retries is retained for backward compatibility with older callers.
    """
    if not text:
        return [0.0] * EMBEDDING_DIMENSIONS

    try:
        model = _get_model()

        # Serializing encode calls is safer with a threaded Gunicorn worker
        # and avoids competing CPU inference calls.
        with _encode_lock:
            vectors = model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        vector = vectors[0].tolist()

        if len(vector) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                "Invalid query embedding dimensions: "
                f"expected {EMBEDDING_DIMENSIONS}, got {len(vector)}"
            )

        return vector

    except Exception as e:
        print(
            "Local embedding failed: "
            f"{type(e).__name__}: {e}"
        )
        return None


def embed_batch(
    texts: list,
    max_retries: int = 1,
) -> list:
    """
    Generate local embeddings for a list of texts.
    """
    if not texts:
        return []

    valid_texts = [
        str(text) if text is not None else ""
        for text in texts
    ]

    try:
        model = _get_model()

        with _encode_lock:
            vectors = model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        embeddings = []

        for vector in vectors:
            vector_list = vector.tolist()

            if len(vector_list) != EMBEDDING_DIMENSIONS:
                raise RuntimeError(
                    "Invalid batch embedding dimensions: "
                    f"expected {EMBEDDING_DIMENSIONS}, "
                    f"got {len(vector_list)}"
                )

            embeddings.append(vector_list)

        return embeddings

    except Exception as e:
        print(
            "Local batch embedding failed: "
            f"{type(e).__name__}: {e}"
        )
        return []


def embed_all_chunks(chunks):
    """
    Compatibility helper.

    Normal production ingestion continues to use ingest_pipeline.py.
    """
    for chunk in chunks:
        chunk["embedding"] = embed_single_text(
            chunk.get("text", "")
        )

    return chunks


if __name__ == "__main__":
    print("=" * 60)
    print("MediCare AI - Local MPNet Query Embedder Test")
    print("=" * 60)

    test_sentence = "What are the symptoms of Type 2 Diabetes?"
    vector = embed_single_text(test_sentence)

    if vector and len(vector) == EMBEDDING_DIMENSIONS:
        print("SUCCESS")
        print(f"Dimensions: {len(vector)}")
        print(f"First 5 values: {vector[:5]}")
    else:
        print("FAILED")
