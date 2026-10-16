"""
Cell-set algebra on the integer grid.

Regions, cutters, swept areas and collision offsets are all sets of unit cells
(or of lattice points) on the same grid, so they all use a single
representation: a boolean mask plus the grid coordinate of its lower-left
entry. Working on the grid rather than with polygons is exact here -- every
coordinate in the problem is an integer and every move is a unit step -- and it
keeps the expensive operations to a handful of vectorized passes.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

from .schemas.instance import PointSequence, PolyominoWithHoles

__all__ = ["CellSet", "dilate", "rasterize", "rasterize_ring"]


@dataclass(frozen=True)
class CellSet:
    """
    A set of unit cells, stored as a boolean mask over a bounding box.

    `mask[row, col]` is the cell at `(origin[0] + col, origin[1] + row)`, so the
    row index runs along `y` and the column index along `x`. A cell `(x, y)`
    stands for the unit square `[x, x + 1] x [y, y + 1]`; the same type is also
    used for sets of lattice *points*, where a cell simply means the point.
    """

    mask: np.ndarray
    origin: tuple[int, int]

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """
        The bounding box as `(x_min, y_min, x_max, y_max)`, with the maximum
        being the last cell that fits, not one past it.
        """
        x, y = self.origin
        return (x, y, x + self.width - 1, y + self.height - 1)

    def __len__(self) -> int:
        return int(self.mask.sum())

    def __contains__(self, cell: tuple[int, int]) -> bool:
        col, row = cell[0] - self.origin[0], cell[1] - self.origin[1]
        if not (0 <= col < self.width and 0 <= row < self.height):
            return False
        return bool(self.mask[row, col])

    def cells(self) -> Iterator[tuple[int, int]]:
        """
        Yields the contained cells in row-major order.
        """
        rows, cols = np.nonzero(self.mask)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            yield (self.origin[0] + col, self.origin[1] + row)

    def translated(self, dx: int, dy: int) -> "CellSet":
        return CellSet(self.mask, (self.origin[0] + dx, self.origin[1] + dy))

    def reflected(self) -> "CellSet":
        """
        The point reflection through the origin, i.e. `{-c for c in self}`.
        """
        _, _, x_max, y_max = self.bounds
        return CellSet(self.mask[::-1, ::-1], (-x_max, -y_max))

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        """
        Vectorized membership test for an `(n, 2)` array of cells; returns a
        boolean array of length `n`. Cells outside the bounding box are simply
        not contained.
        """
        col = points[:, 0] - self.origin[0]
        row = points[:, 1] - self.origin[1]
        inside = (col >= 0) & (col < self.width) & (row >= 0) & (row < self.height)
        result = np.zeros(len(points), dtype=bool)
        result[inside] = self.mask[row[inside], col[inside]]
        return result

    def __and__(self, other: "CellSet") -> "CellSet":
        return _combine(self, other, np.logical_and)

    def __or__(self, other: "CellSet") -> "CellSet":
        return _combine(self, other, np.logical_or)

    def difference(self, other: "CellSet") -> "CellSet":
        """
        The cells of `self` that are not in `other`.
        """
        aligned = np.zeros_like(self.mask)
        _blit(aligned, self.origin, other.mask, other.origin)
        return CellSet(self.mask & ~aligned, self.origin)

    def to_polygons(self):
        """
        The cells as a shapely geometry, for error messages and plotting.

        Only meant for small sets; a full swept area is far cheaper to handle
        as a mask.
        """
        return unary_union([box(x, y, x + 1, y + 1) for x, y in self.cells()])


def _combine(a: CellSet, b: CellSet, op) -> CellSet:
    """
    Applies a binary boolean operation to two cell sets over the union of their
    bounding boxes.
    """
    x_min = min(a.origin[0], b.origin[0])
    y_min = min(a.origin[1], b.origin[1])
    x_max = max(a.bounds[2], b.bounds[2])
    y_max = max(a.bounds[3], b.bounds[3])
    shape = (y_max - y_min + 1, x_max - x_min + 1)
    left = np.zeros(shape, dtype=bool)
    right = np.zeros(shape, dtype=bool)
    _blit(left, (x_min, y_min), a.mask, a.origin)
    _blit(right, (x_min, y_min), b.mask, b.origin)
    return CellSet(op(left, right), (x_min, y_min))


def _blit(
    target: np.ndarray,
    target_origin: tuple[int, int],
    source: np.ndarray,
    source_origin: tuple[int, int],
) -> None:
    """
    ORs `source` into `target`, clipping whatever falls outside.
    """
    dx = source_origin[0] - target_origin[0]
    dy = source_origin[1] - target_origin[1]
    height, width = source.shape
    rows = slice(max(dy, 0), min(dy + height, target.shape[0]))
    cols = slice(max(dx, 0), min(dx + width, target.shape[1]))
    if rows.start >= rows.stop or cols.start >= cols.stop:
        return
    target[rows, cols] |= source[
        rows.start - dy : rows.stop - dy, cols.start - dx : cols.stop - dx
    ]


def rasterize_ring(ring: PointSequence) -> CellSet:
    """
    The unit cells enclosed by a single closed boundary.
    """
    return _rasterize([ring])


def rasterize(polyomino: PolyominoWithHoles) -> CellSet:
    """
    The unit cells of a polyomino, holes excluded.
    """
    return _rasterize([polyomino.outer_boundary, *polyomino.inner_boundaries])


def _rasterize(rings: Sequence[PointSequence]) -> CellSet:
    """
    Scan-converts a set of closed rectilinear boundaries by counting, for every
    cell, how often a ray to its west crosses a vertical edge; odd means inside.

    Feeding the holes through the same parity count is what makes them work
    without any special casing, and it does not care about orientation.
    """
    x_min = min(min(ring.x) for ring in rings)
    y_min = min(min(ring.y) for ring in rings)
    x_max = max(max(ring.x) for ring in rings)
    y_max = max(max(ring.y) for ring in rings)
    width, height = x_max - x_min, y_max - y_min

    # One column per vertical grid line, so that a prefix XOR over the columns
    # gives the parity of the crossings west of each cell.
    crossings = np.zeros((height, width + 1), dtype=np.uint8)
    for ring in rings:
        points = list(zip(ring.x, ring.y, strict=True))
        for (x1, y1), (_, y2) in zip(points, [*points[1:], points[0]], strict=True):
            if y1 == y2:
                continue  # horizontal edges are never crossed by a westward ray
            low, high = min(y1, y2), max(y1, y2)
            crossings[low - y_min : high - y_min, x1 - x_min] ^= 1

    np.bitwise_xor.accumulate(crossings, axis=1, out=crossings)
    return CellSet(crossings[:, :width].view(bool), (x_min, y_min))


def _horizontal_runs(cells: CellSet) -> Iterator[tuple[int, int, int]]:
    """
    Yields the maximal horizontal runs of `cells` as `(x, y, length)`.

    Dilating by whole runs instead of by individual cells is what keeps the
    sweep cost proportional to the number of runs of the cutter rather than to
    its area.
    """
    padded = np.zeros((cells.height, cells.width + 2), dtype=bool)
    padded[:, 1:-1] = cells.mask
    starts = np.nonzero(padded[:, 1:-1] & ~padded[:, :-2])
    ends = np.nonzero(padded[:, 1:-1] & ~padded[:, 2:])
    for row, start_col, end_col in zip(
        starts[0].tolist(), starts[1].tolist(), ends[1].tolist(), strict=True
    ):
        yield (
            cells.origin[0] + start_col,
            cells.origin[1] + row,
            end_col - start_col + 1,
        )


def _dilate_horizontally(mask: np.ndarray, length: int) -> np.ndarray:
    """
    The union of `mask` shifted east by `0 .. length - 1`, via a prefix sum so
    that the cost does not depend on `length`.
    """
    height, width = mask.shape
    prefix = np.zeros((height, width + 1), dtype=np.int32)
    np.cumsum(mask, axis=1, dtype=np.int32, out=prefix[:, 1:])
    columns = np.arange(width + length - 1)
    high = np.minimum(columns + 1, width)
    low = np.maximum(columns - length + 1, 0)
    return prefix[:, high] - prefix[:, low] > 0


def dilate(cells: CellSet, structuring: CellSet) -> CellSet:
    """
    The Minkowski sum `{c + s for c in cells for s in structuring}`.
    """
    x_min = cells.origin[0] + structuring.origin[0]
    y_min = cells.origin[1] + structuring.origin[1]
    shape = (
        cells.height + structuring.height - 1,
        cells.width + structuring.width - 1,
    )
    result = np.zeros(shape, dtype=bool)
    for x, y, length in _horizontal_runs(structuring):
        swept = _dilate_horizontally(cells.mask, length)
        row = y - structuring.origin[1]
        col = x - structuring.origin[0]
        result[row : row + cells.height, col : col + swept.shape[1]] |= swept
    return CellSet(result, (x_min, y_min))
