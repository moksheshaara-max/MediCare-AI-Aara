"""MediCare AI 768-D MPNet query embeddings.

Local development/ingestion retains the original SentenceTransformer CPU backend.
Render can instead run the *same MPNet model* as quantized ONNX on CPU,
without importing PyTorch or Sentence Transformers into the web-service process.

MEDICARE_EMBEDDER_BACKEND=auto (default): choose ONNX when model artifacts exist;
otherwise use the original PyTorch backend for local development.
MEDICARE_EMBEDDER_BACKEND=onnx: explicitly require the low-memory backend.
MEDICARE_EMBEDDER_BACKEND=torch: explicitly use the original backend.

The original SentenceTransformer has Transformer -> mean pooling -> L2 Normalize
modules; setting normalize_embeddings=False on encode does not remove the module's
Normalize layer. ONNX reproduces those steps, with some quantization variance.
"""

import os
import threading
from pathlib import Path
from typing import List, Optional

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIMENSIONS = 768
MAX_SEQ_LENGTH = 384
TORCH_CPU_THREADS = int(os.getenv("TORCH_CPU_THREADS", "2"))

_ROOT = Path(__file__).resolve().parents[1]
_ASSET_DIR = Path(os.getenv("MEDICARE_MPNet_ASSET_DIR", str(_ROOT / ".runtime_models")))
_ONNX_FILE = _ASSET_DIR / "onnx" / "model_quint8_avx2.onnx"
_TOKENIZER_FILE = _ASSET_DIR / "tokenizer.json"

_model = None  # Legacy PyTorch SentenceTransformer cache
_onnx_session = None
_onnx_tokenizer = None
_init_lock = threading.Lock()
_encode_lock = threading.Lock()


def _backend() -> str:
    value = os.getenv("MEDICARE_EMBEDDER_BACKEND", "auto").strip().lower()
    if value not in {"auto", "onnx", "torch"}:
        raise ValueError("Invalid MEDICARE_EMBEDDER_BACKEND; choose auto, onnx or torch")
    if value == "auto":
        return "onnx" if _ONNX_FILE.is_file() and _TOKENIZER_FILE.is_file() else "torch"
    return value


def _get_model():
    """Load original PyTorch model lazily (local-dev backward compatibility)."""
    global _model
    if _model is not None:
        return _model
    with _init_lock:
        if _model is None:
            # Never import either heavy library on a Render ONNX worker.
            import torch
            from sentence_transformers import SentenceTransformer
            torch.set_num_threads(TORCH_CPU_THREADS)
            print(f"Loading local PyTorch MPNet: {MODEL_ID}", flush=True)
            model = SentenceTransformer(MODEL_ID, device="cpu")
            dim = model.get_sentence_embedding_dimension()
            if dim != EMBEDDING_DIMENSIONS:
                raise RuntimeError(f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {dim}")
            _model = model
    return _model


def _get_onnx_session():
    """Load quantized MPNet only on its first query, one session per worker."""
    global _onnx_session, _onnx_tokenizer
    if _onnx_session is not None:
        return _onnx_session, _onnx_tokenizer
    with _init_lock:
        if _onnx_session is None:
            if not _ONNX_FILE.is_file() or not _TOKENIZER_FILE.is_file():
                raise FileNotFoundError(
                    f"ONNX MPNet assets missing from {_ASSET_DIR}. "
                    "Run python scripts/download_render_mpnet.py in the build command."
                )
            import onnxruntime as ort
            from tokenizers import Tokenizer
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            print("Loading 8-bit ONNX MPNet on CPU", flush=True)
            session = ort.InferenceSession(
                str(_ONNX_FILE), sess_options=options,
                providers=["CPUExecutionProvider"]
            )
            tokenizer = Tokenizer.from_file(str(_TOKENIZER_FILE))
            tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)
            _onnx_tokenizer = tokenizer
            _onnx_session = session
            print("ONNX MPNet ready (768-D)", flush=True)
    return _onnx_session, _onnx_tokenizer


def _encode_onnx_one(text: str) -> List[float]:
    import numpy as np

    session, tokenizer = _get_onnx_session()
    encoded = tokenizer.encode(text)
    input_ids = np.asarray([encoded.ids], dtype=np.int64)
    attention_mask = np.asarray([encoded.attention_mask], dtype=np.int64)

    feed = {}
    for input_def in session.get_inputs():
        key = input_def.name
        if key == "input_ids":
            feed[key] = input_ids
        elif key == "attention_mask":
            feed[key] = attention_mask
        elif key == "token_type_ids":
            feed[key] = np.zeros_like(input_ids)
        elif key == "position_ids":
            feed[key] = np.arange(input_ids.shape[1], dtype=np.int64)[None, :]
        else:
            raise RuntimeError(f"Unrecognized ONNX MPNet input: {key}")

    names = [output.name for output in session.get_outputs()]
    states = session.run(None, feed)
    # The exported sentence-transformers transformer produces a 3-D
    # last_hidden_state, not already-pooled sentence embeddings.
    idx = names.index("last_hidden_state") if "last_hidden_state" in names else 0
    hidden = np.asarray(states[idx], dtype=np.float32)
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != EMBEDDING_DIMENSIONS:
        raise RuntimeError(f"Unexpected ONNX hidden-state shape: {hidden.shape}")

    # Sentence Transformers' mean-token pooling followed by Normalize module.
    weights = attention_mask.astype(np.float32)[:, :, None]
    sentence = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
    sentence = sentence[0]
    length = float(np.linalg.norm(sentence))
    if not np.isfinite(length) or length < 1e-12:
        raise RuntimeError("Invalid zero or non-finite ONNX embedding")
    sentence = sentence / length
    return [float(x) for x in sentence.tolist()]


def embed_single_text(text: str, max_retries: int = 1) -> Optional[List[float]]:
    """Query embedding; returns None on failure (never silently swaps models)."""
    if not text:
        return [0.0] * EMBEDDING_DIMENSIONS
    try:
        backend = _backend()
        with _encode_lock:
            if backend == "onnx":
                vector = _encode_onnx_one(str(text))
            else:
                vectors = _get_model().encode(
                    [str(text)], convert_to_numpy=True,
                    normalize_embeddings=False, show_progress_bar=False
                )
                vector = [float(v) for v in vectors[0].tolist()]
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(f"Expected 768 embedding dimensions, got {len(vector)}")
        return vector
    except Exception as exc:
        print(f"MPNet embedding failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def embed_batch(texts: list, max_retries: int = 1) -> list:
    """Conservative sequential encoding bounds inference peak memory."""
    if not texts:
        return []
    results = []
    for text in texts:
        vector = embed_single_text(str(text) if text is not None else "")
        if vector is None:
            return []
        results.append(vector)
    return results


def embed_all_chunks(chunks):
    """Compatibility helper. Normal ingestion uses ingest_pipeline.py."""
    for chunk in chunks:
        chunk["embedding"] = embed_single_text(chunk.get("text", ""))
    return chunks


if __name__ == "__main__":
    test_vector = embed_single_text("What are the symptoms of Type 2 Diabetes?")
    if test_vector is None:
        raise SystemExit("Embedding self-test failed")
    print(f"Embedding self-test: {len(test_vector)} dimensions")
