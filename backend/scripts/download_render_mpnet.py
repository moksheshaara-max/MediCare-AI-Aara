"""Build-time model download for Render's free 512 MiB CPU instance.
Run from backend root *after* installing requirements-render.txt.
Weights are not committed to Git and need not exist during image build.
"""
from hashlib import sha256
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
MODEL_PATH = "onnx/model_quint8_avx2.onnx"
EXPECTED_SHA256 = "aa5c27172d77bbd1cbae3628cbac4b26d7c12adabff25d2d4285d0f29159b237"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / ".runtime_models"


def main():
    ASSETS.mkdir(exist_ok=True)
    for filename in ("tokenizer.json", MODEL_PATH):
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            local_dir=str(ASSETS),
        )
        print(f"Downloaded: {path}", flush=True)
    model_path = ASSETS / MODEL_PATH
    with model_path.open("rb") as model:
        digest = sha256()
        for chunk in iter(lambda: model.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("Quantized MPNet checksum mismatch; refusing startup")
    if not (ASSETS / "tokenizer.json").is_file():
        raise RuntimeError("Tokenizer missing")
    print(f"MPNet ready for Render: {model_path.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
