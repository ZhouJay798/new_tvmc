"""Model archives, file readout and report serialization."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_STORED

from tvmc.robust_transforms import EvaluationModel


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pack(directory, target):
    with ZipFile(target, "w", compression=ZIP_STORED) as archive:
        for item in sorted(directory.iterdir()):
            if item.is_file():
                archive.write(item, item.name)


def reader(parameters, scratch):
    def read(artifact):
        with TemporaryDirectory(dir=scratch) as temp, ZipFile(artifact) as archive:
            for name in archive.namelist():
                if Path(name).name != name or name.startswith("."):
                    raise ValueError("Invalid model archive member")
            archive.extractall(temp)
            lab = EvaluationModel(Path(temp))
            lab.repeats = parameters["repeats"]
            word, _ = lab.read(parameters["probes"], delta=parameters.get("delta", 1.0))
            return word.tolist()
    return read
