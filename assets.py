"""Pinned public assets for the file experiment. Run from the project root."""
from pathlib import Path
from hashlib import file_digest

from huggingface_hub import HfApi, snapshot_download
from prepare_data import main as prepare_corpus

REPO = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
MODEL = Path("models") / REPO.split("/")[1] / REVISION
CORPUS = Path(".cache/public_data/tinyshakespeare-6f9487a.txt")
FILES = ["model.safetensors", "config.json", "generation_config.json", "merges.txt",
         "tokenizer.json", "tokenizer_config.json", "vocab.json"]


def prepare():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    weight = next(item for item in info.siblings if item.rfilename == FILES[0])
    snapshot_download(REPO, revision=REVISION, local_dir=MODEL, allow_patterns=FILES, token=False)
    for name in FILES:
        if not (MODEL / name).is_file():
            raise FileNotFoundError(MODEL / name)
    with (MODEL / FILES[0]).open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != weight.lfs.sha256:
            raise ValueError("Weight checksum differs from the pinned official revision")
    prepare_corpus()


if __name__ == "__main__":
    prepare()
