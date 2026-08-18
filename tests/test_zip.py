import io
import zipfile
from pathlib import Path

import pytest

from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution
from cgshop2027_pyutils.zip import (
    BadSolutionFile,
    ZipSolutionIterator,
    ZipWriter,
)
from cgshop2027_pyutils.zip.zip_reader_errors import (
    FileTooLargeError,
    InvalidFileName,
    InvalidZipError,
    NoSolutionsError,
    ZipTooLargeError,
)

REGION = {
    "outer_boundary": {"x": [0, 3, 3, 0], "y": [0, 0, 3, 3]},
    "inner_boundaries": [],
}
CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}
SWEEP = [("E", 2), ("N", 1), ("W", 2), ("N", 1), ("E", 2), ("W", 2), ("S", 2)]


def make_instance(uid: str = "demo") -> CGSHOP2027Instance:
    return CGSHOP2027Instance(
        instance_uid=uid,
        region_to_cover=REGION,
        cutter=CUTTER,
        cutter_center=(0, 0),
        number_of_cutters=2,
    )


def make_solution(uid: str = "demo") -> CGSHOP2027Solution:
    return CGSHOP2027Solution(
        instance_uid=uid, cutters=[{"start": (0, 0), "steps": SWEEP}]
    )


def write_solutions(path: Path, *uids: str) -> Path:
    with ZipWriter(path) as writer:
        for uid in uids:
            writer.add_solution(make_solution(uid))
    return path


def write_raw(path: Path, members: dict[str, str]) -> Path:
    """Builds an archive from literal member names and contents."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


# --------------------------------------------------------------------------
# ZipWriter
# --------------------------------------------------------------------------


def test_writer_names_members_after_the_instance_uid(tmp_path: Path):
    path = tmp_path / "bundle.zip"
    with ZipWriter(path) as writer:
        writer.add_instance(make_instance("alpha"))
        writer.add_solution(make_solution("beta"))
    with zipfile.ZipFile(path) as archive:
        assert sorted(archive.namelist()) == [
            "alpha.instance.json",
            "beta.solution.json",
        ]


def test_writer_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "bundle.zip"
    path.write_text("not a zip")
    with pytest.raises(FileExistsError):
        ZipWriter(path)


def test_writer_closes_the_archive_on_exit(tmp_path: Path):
    path = tmp_path / "bundle.zip"
    with ZipWriter(path) as writer:
        writer.add_solution(make_solution())
    assert zipfile.is_zipfile(path)


# --------------------------------------------------------------------------
# ZipSolutionIterator: reading
# --------------------------------------------------------------------------


def test_solutions_survive_a_zip_round_trip(tmp_path: Path):
    originals = {uid: make_solution(uid) for uid in ("alpha", "beta")}
    path = write_solutions(tmp_path / "solutions.zip", *originals)
    read_back = list(ZipSolutionIterator(path))
    assert len(read_back) == 2
    for solution in read_back:
        expected = originals[solution.instance_uid]
        assert solution.cutters == expected.cutters
        assert solution.content_type == expected.content_type


def test_the_origin_of_each_solution_is_recorded_in_its_meta(tmp_path: Path):
    path = write_solutions(tmp_path / "solutions.zip", "alpha")
    (solution,) = ZipSolutionIterator(path)
    assert solution.meta["zip_info"] == {
        "zip_file": str(path),
        "file_in_zip": "alpha.solution.json",
    }


def test_the_short_extension_is_accepted(tmp_path: Path):
    path = write_raw(
        tmp_path / "solutions.zip",
        {"alpha.sol.json": make_solution().model_dump_json()},
    )
    assert len(list(ZipSolutionIterator(path))) == 1


def test_solutions_are_read_from_a_file_object(tmp_path: Path):
    path = write_solutions(tmp_path / "solutions.zip", "alpha")
    handle = io.BytesIO(path.read_bytes())
    assert len(list(ZipSolutionIterator(handle))) == 1


def test_solutions_in_subfolders_are_found(tmp_path: Path):
    path = write_raw(
        tmp_path / "solutions.zip",
        {"runs/best/alpha.solution.json": make_solution().model_dump_json()},
    )
    assert len(list(ZipSolutionIterator(path))) == 1


@pytest.mark.parametrize("folder", ["__MACOSX", ".hidden"])
def test_hidden_folders_are_skipped(tmp_path: Path, folder: str):
    path = write_raw(
        tmp_path / "solutions.zip",
        {f"{folder}/alpha.solution.json": make_solution().model_dump_json()},
    )
    with pytest.raises(NoSolutionsError):
        list(ZipSolutionIterator(path))


def test_an_archive_without_solutions_is_rejected(tmp_path: Path):
    path = tmp_path / "instances.zip"
    with ZipWriter(path) as writer:
        writer.add_instance(make_instance())
    with pytest.raises(NoSolutionsError):
        list(ZipSolutionIterator(path))


# --------------------------------------------------------------------------
# ZipSolutionIterator: rejection
# --------------------------------------------------------------------------


def test_a_malformed_solution_names_its_file(tmp_path: Path):
    path = write_raw(tmp_path / "solutions.zip", {"alpha.solution.json": "{not json"})
    with pytest.raises(BadSolutionFile) as exc_info:
        list(ZipSolutionIterator(path))
    assert exc_info.value.file_name == "alpha.solution.json"
    assert "alpha.solution.json" in str(exc_info.value)


def test_a_solution_violating_the_schema_is_rejected(tmp_path: Path):
    """A well-formed JSON document still has to satisfy the solution schema."""
    path = write_raw(
        tmp_path / "solutions.zip",
        {"alpha.solution.json": '{"instance_uid": "alpha", "cutters": [[1, 2]]}'},
    )
    with pytest.raises(BadSolutionFile):
        list(ZipSolutionIterator(path))


@pytest.mark.parametrize(
    "name", ["../evil.solution.json", "/abs/evil.solution.json", "a/../../evil.json"]
)
def test_path_traversal_is_rejected(tmp_path: Path, name: str):
    path = write_raw(tmp_path / "solutions.zip", {name: "{}"})
    with pytest.raises(InvalidFileName):
        list(ZipSolutionIterator(path))


def test_an_oversized_member_is_rejected(tmp_path: Path):
    path = write_solutions(tmp_path / "solutions.zip", "alpha")
    with pytest.raises(FileTooLargeError):
        list(ZipSolutionIterator(path, file_size_limit=10))


def test_an_oversized_archive_is_rejected(tmp_path: Path):
    path = write_solutions(tmp_path / "solutions.zip", "alpha", "beta")
    with pytest.raises(ZipTooLargeError):
        # Generous per file, so that only the total trips the limit.
        list(ZipSolutionIterator(path, file_size_limit=10**9, zip_size_limit=10))


def test_something_that_is_not_a_zip_is_rejected(tmp_path: Path):
    path = tmp_path / "solutions.zip"
    path.write_text("this is not an archive")
    with pytest.raises(InvalidZipError):
        list(ZipSolutionIterator(path))


def test_a_corrupted_member_is_rejected(tmp_path: Path):
    """
    Members are stored uncompressed, so flipping a byte inside the payload
    leaves the archive structure intact and only breaks the CRC.
    """
    path = write_solutions(tmp_path / "solutions.zip", "alpha")
    raw = bytearray(path.read_bytes())
    marker = b'"instance_uid":"alpha"'
    offset = raw.index(marker) + marker.index(b"alpha")
    raw[offset : offset + 5] = b"gamma"
    path.write_bytes(raw)
    with pytest.raises(InvalidZipError, match="CRC"):
        list(ZipSolutionIterator(path))
