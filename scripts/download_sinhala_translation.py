"""Fetch a pinned official Indic-to-English model; verify LFS checksums."""
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_sinhala_ocr import download, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
REPO = "Helsinki-NLP/opus-mt-inc-en"
REV = "b237db6975a3f7973fb149adc7166bfa30f8f224"
FILES = ("config.json", "generation_config.json", "pytorch_model.bin", "source.spm",
         "target.spm", "tokenizer_config.json", "vocab.json", "README.md")


def main():
    destination = ROOT / "models/sinhala_translation"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true", timeout=60) as response:
        metadata = json.load(response)
    files = {item["rfilename"]: item for item in metadata["siblings"]}
    provenance = {"repository": REPO, "revision": REV, "license": "Apache-2.0", "files": {}}
    for name in FILES:
        path = destination / name
        checksum = files[name].get("lfs", {}).get("sha256")
        download(f"https://huggingface.co/{REPO}/resolve/{REV}/{name}", path, checksum)
        if path.stat().st_size != files[name]["size"]:
            raise ValueError("Unexpected file size: " + name)
        provenance["files"][name] = {"size": path.stat().st_size, "sha256": sha256(path)}
    write_json(destination / "provenance.json", provenance)


if __name__ == "__main__":
    main()
