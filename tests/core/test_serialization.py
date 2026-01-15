from dataclasses import dataclass
from pathlib import Path

from qubex.core.serialization import SerializableModel


@dataclass
class PathModel(SerializableModel):
    p: Path


def test_path_subclass_serialization():
    # PosixPath or WindowsPath depending on OS, both are Path subclasses
    p = Path("/tmp/foo")
    m = PathModel(p=p)
    json_str = m.to_json()
    # Check that it serialized (didn't raise CodecError)
    assert "/tmp/foo" in json_str.replace("\\", "/")

    m2 = PathModel.from_json(json_str)
    assert m2.p == p
