import io
from pathlib import Path

from cgshop2027_pyutils.io import read_instance, read_solution
from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution

SQUARE_REGION = {
    "outer_boundary": {"x": [0, 4, 4, 0], "y": [0, 0, 4, 4]},
    "inner_boundaries": [],
}
SQUARE_CUTTER = {"x": [0, 2, 2, 0], "y": [0, 0, 2, 2]}
CUTTER_TOUR = {"x": [1, 3, 3, 1], "y": [1, 1, 3, 3]}


def test_read_instance_from_path(tmp_path: Path):
    instance = CGSHOP2027Instance(
        instance_uid="demo",
        region_to_cover=SQUARE_REGION,
        cutter=SQUARE_CUTTER,
        cutter_center=(1, 1),
        number_of_cutters=3,
    )
    path = tmp_path / "demo.instance.json"
    path.write_text(instance.model_dump_json())
    assert read_instance(path) == instance


def test_read_solution_from_file_object():
    solution = CGSHOP2027Solution(
        instance_uid="demo", tours=[CUTTER_TOUR], meta={"author": "you"}
    )
    assert read_solution(io.StringIO(solution.model_dump_json())) == solution
