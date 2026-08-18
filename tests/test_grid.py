import numpy as np
import pytest
from shapely.geometry import Polygon

from cgshop2027_pyutils.grid import (
    CellSet,
    dilate,
    rasterize,
    rasterize_ring,
    self_offsets,
)
from cgshop2027_pyutils.schemas.instance import PointSequence, PolyominoWithHoles

SQUARE = {"x": [0, 5, 5, 0], "y": [0, 0, 5, 5]}
HOLE = {"x": [1, 1, 3, 3], "y": [1, 3, 3, 1]}
L_SHAPE = {"x": [0, 3, 3, 1, 1, 0], "y": [0, 0, 1, 1, 3, 3]}
STAIRCASE = {
    "x": [0, 3, 3, 2, 2, 1, 1, 0],
    "y": [0, 0, 1, 1, 2, 2, 3, 3],
}


def ring(spec: dict[str, list[int]]) -> PointSequence:
    return PointSequence(**spec)


def shapely_cells(outer, holes=()):
    """The cell set of a boundary, decided one cell at a time via shapely."""
    polygon = Polygon(
        list(zip(outer["x"], outer["y"], strict=True)),
        holes=[list(zip(h["x"], h["y"], strict=True)) for h in holes],
    )
    x_min, y_min, x_max, y_max = (int(v) for v in polygon.bounds)
    return {
        (x, y)
        for x in range(x_min - 1, x_max + 1)
        for y in range(y_min - 1, y_max + 1)
        if polygon.contains(Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]))
    }


# --------------------------------------------------------------------------
# CellSet
# --------------------------------------------------------------------------


def test_cell_set_membership_and_bounds():
    cells = CellSet(np.array([[True, False], [True, True]]), (4, 7))
    assert cells.bounds == (4, 7, 5, 8)
    assert len(cells) == 3
    assert (4, 7) in cells
    assert (5, 7) not in cells
    assert (99, 99) not in cells
    assert sorted(cells.cells()) == [(4, 7), (4, 8), (5, 8)]


def test_translating_moves_every_cell():
    cells = CellSet(np.array([[True, False]]), (0, 0))
    assert sorted(cells.translated(3, -2).cells()) == [(3, -2)]


def test_reflecting_negates_every_cell():
    cells = CellSet(np.array([[True, True], [False, True]]), (1, 1))
    assert sorted(cells.reflected().cells()) == sorted(
        (-x, -y) for x, y in cells.cells()
    )


def test_set_operations_align_differing_origins():
    left = CellSet(np.array([[True, True]]), (0, 0))
    right = CellSet(np.array([[True, True]]), (1, 0))
    assert sorted((left | right).cells()) == [(0, 0), (1, 0), (2, 0)]
    assert sorted((left & right).cells()) == [(1, 0)]
    assert sorted(left.difference(right).cells()) == [(0, 0)]


def test_contains_points_matches_membership():
    cells = CellSet(np.array([[True, False], [False, True]]), (-1, -1))
    points = np.array([(-1, -1), (0, -1), (-1, 0), (0, 0), (50, 50)])
    expected = [point in cells for point in map(tuple, points.tolist())]
    assert list(cells.contains_points(points)) == expected


# --------------------------------------------------------------------------
# rasterization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary", [SQUARE, L_SHAPE, STAIRCASE], ids=["square", "L", "staircase"]
)
def test_rasterize_ring_agrees_with_shapely(boundary: dict[str, list[int]]):
    """
    The scan conversion replaces a library routine, so it is checked cell by
    cell against shapely rather than against hand-written expectations.
    """
    assert set(rasterize_ring(ring(boundary)).cells()) == shapely_cells(boundary)


def test_rasterize_excludes_holes():
    polyomino = PolyominoWithHoles(outer_boundary=SQUARE, inner_boundaries=[HOLE])
    assert set(rasterize(polyomino).cells()) == shapely_cells(SQUARE, [HOLE])


def test_rasterize_handles_several_holes():
    holes = [
        {"x": [1, 1, 2, 2], "y": [1, 2, 2, 1]},
        {"x": [3, 3, 4, 4], "y": [3, 4, 4, 3]},
    ]
    polyomino = PolyominoWithHoles(outer_boundary=SQUARE, inner_boundaries=holes)
    assert set(rasterize(polyomino).cells()) == shapely_cells(SQUARE, holes)


def test_rasterize_is_translation_invariant():
    shifted = {"x": [x + 7 for x in L_SHAPE["x"]], "y": [y - 4 for y in L_SHAPE["y"]]}
    assert sorted(rasterize_ring(ring(shifted)).cells()) == sorted(
        (x + 7, y - 4) for x, y in rasterize_ring(ring(L_SHAPE)).cells()
    )


# --------------------------------------------------------------------------
# Minkowski sums
# --------------------------------------------------------------------------


def naive_dilate(cells: CellSet, structuring: CellSet) -> set[tuple[int, int]]:
    return {
        (ax + bx, ay + by) for ax, ay in cells.cells() for bx, by in structuring.cells()
    }


def random_cell_set(rng, shape) -> CellSet:
    origin = (int(rng.integers(-3, 4)), int(rng.integers(-3, 4)))
    return CellSet(rng.random(shape) < 0.4, origin)


def test_dilate_by_a_single_cell_is_a_translation():
    cells = rasterize_ring(ring(L_SHAPE))
    point = CellSet(np.array([[True]]), (2, -3))
    assert sorted(dilate(cells, point).cells()) == sorted(
        cells.translated(2, -3).cells()
    )


def test_dilate_matches_the_naive_sum():
    """
    `dilate` works run by run rather than cell by cell, so it is checked
    against the obvious double loop on random inputs.
    """
    rng = np.random.default_rng(0)
    for _ in range(100):
        cells = random_cell_set(rng, (5, 6))
        structuring = random_cell_set(rng, (3, 4))
        assert set(dilate(cells, structuring).cells()) == naive_dilate(
            cells, structuring
        )


def test_self_offsets_matches_the_naive_difference_set():
    rng = np.random.default_rng(1)
    for _ in range(100):
        cells = random_cell_set(rng, (4, 5))
        expected = {
            (ax - bx, ay - by) for ax, ay in cells.cells() for bx, by in cells.cells()
        }
        assert set(self_offsets(cells).cells()) == expected


def test_self_offsets_of_a_single_cell_is_the_zero_offset():
    assert sorted(self_offsets(CellSet(np.array([[True]]), (5, 5))).cells()) == [(0, 0)]


def test_self_offsets_is_symmetric():
    offsets = self_offsets(rasterize_ring(ring(L_SHAPE)))
    assert sorted(offsets.cells()) == sorted((-x, -y) for x, y in offsets.cells())
