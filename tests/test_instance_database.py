from pathlib import Path

import pytest

from cgshop2027_pyutils.instance_database import InstanceDatabase
from cgshop2027_pyutils.instance_database.instance_file_database import (
    InstanceFileDatabase,
)
from cgshop2027_pyutils.instance_database.instance_zip_database import (
    InstanceZipDatabase,
)
from cgshop2027_pyutils.schemas import CGSHOP2027Instance
from cgshop2027_pyutils.zip import ZipWriter

REGION = {
    "outer_boundary": {"x": [0, 3, 3, 0], "y": [0, 0, 3, 3]},
    "inner_boundaries": [],
}
CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}


def make_instance(uid: str) -> CGSHOP2027Instance:
    return CGSHOP2027Instance(
        instance_uid=uid,
        region_to_cover=REGION,
        cutter=CUTTER,
        cutter_center=(0, 0),
        number_of_cutters=2,
    )


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A folder holding `alpha` at the top level and `beta` in a subfolder."""
    root = tmp_path / "instances"
    (root / "nested").mkdir(parents=True)
    (root / "alpha.instance.json").write_text(make_instance("alpha").model_dump_json())
    (root / "nested" / "beta.instance.json").write_text(
        make_instance("beta").model_dump_json()
    )
    return root


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "instances.zip"
    with ZipWriter(path) as writer:
        for uid in ("alpha", "beta"):
            writer.add_instance(make_instance(uid))
    return path


@pytest.fixture(params=["folder", "archive"])
def database(request) -> InstanceDatabase:
    """The same content, once as a folder and once as an archive."""
    return InstanceDatabase(request.getfixturevalue(request.param))


# --------------------------------------------------------------------------
# behavior shared by both backends
# --------------------------------------------------------------------------


def test_iterating_yields_every_instance(database: InstanceDatabase):
    assert sorted(instance.instance_uid for instance in database) == ["alpha", "beta"]


@pytest.mark.parametrize(
    "key",
    [
        "alpha",
        "alpha.json",
        "alpha.instance.json",
        "some/where/alpha",
        "some/where/alpha.instance.json",
    ],
)
def test_lookup_accepts_any_spelling_of_the_name(database: InstanceDatabase, key: str):
    """
    A key is reduced the same way an instance name is derived from a file, so
    the documented file name `NAME.instance.json` works as a key, as does a
    path to it or the bare name.
    """
    assert database[key].instance_uid == "alpha"


def test_lookup_of_an_unknown_instance_raises(database: InstanceDatabase):
    with pytest.raises(KeyError):
        database["gamma"]


def test_instances_are_fully_parsed(database: InstanceDatabase):
    instance = database["alpha"]
    assert instance.number_of_cutters == 2
    assert instance.region_to_cover.outer_boundary.x == [0, 3, 3, 0]


# --------------------------------------------------------------------------
# backend selection and caching
# --------------------------------------------------------------------------


def test_a_folder_selects_the_file_backend(folder: Path):
    assert isinstance(InstanceDatabase(folder)._inner_database, InstanceFileDatabase)


def test_an_archive_selects_the_zip_backend(archive: Path):
    assert isinstance(InstanceDatabase(archive)._inner_database, InstanceZipDatabase)


def test_a_missing_path_is_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        InstanceDatabase(tmp_path / "nowhere")


def test_a_file_that_is_not_an_archive_is_rejected(tmp_path: Path):
    path = tmp_path / "instances.zip"
    path.write_text("not an archive")
    with pytest.raises(FileNotFoundError, match="not a valid zipfile"):
        InstanceDatabase(path)


def test_the_backends_reject_a_missing_path_directly(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        InstanceFileDatabase(tmp_path / "nowhere")


def test_caching_returns_the_same_object(folder: Path, archive: Path):
    for path in (folder, archive):
        database = InstanceDatabase(path, enable_cache=True)
        assert database["alpha"] is database["alpha"]


def test_without_caching_each_lookup_reparses(folder: Path):
    database = InstanceDatabase(folder)
    first, second = database["alpha"], database["alpha"]
    assert first is not second
    assert first == second


# --------------------------------------------------------------------------
# folder-specific behavior
# --------------------------------------------------------------------------


def test_instances_in_subfolders_are_found(folder: Path):
    assert InstanceDatabase(folder)["beta"].instance_uid == "beta"


@pytest.mark.parametrize("hidden", ["__MACOSX", ".hidden"])
def test_hidden_folders_are_skipped(folder: Path, hidden: str):
    (folder / hidden).mkdir()
    (folder / hidden / "gamma.instance.json").write_text(
        make_instance("gamma").model_dump_json()
    )
    database = InstanceDatabase(folder)
    assert sorted(instance.instance_uid for instance in database) == ["alpha", "beta"]
    with pytest.raises(KeyError):
        database["gamma"]
