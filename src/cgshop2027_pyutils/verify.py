"""
Verification of a solution against its instance.

The checks that need only the solution itself (well-formed steps, every cutter
returning to its start) live in the schema. What is left needs both objects:
the cutters must never overlap each other, and together they must sweep the
whole region.

Both checks avoid walking the trajectories cell by cell:

* The swept area is `cutter (+) anchors`, the Minkowski sum of the cutter with
  the set of all anchor positions any cutter ever visits. Unioning the anchors
  first makes revisits free, and one dilation replaces one placement per cutter
  per time step. Continuous motion adds nothing to sweep, because for a union
  of unit cells `C` and a unit vector `e` we have
  `C (+) [p, p+e] = (C+p) | (C+p+e)`.
* Two placements of the cutter overlap in area exactly when the offset between
  them lies in `cutter (+) (-cutter)`, precomputed once per instance. Overlap
  during a move is then pure case analysis on the relative velocity, so
  continuous collisions also reduce to lattice lookups.
"""

from itertools import pairwise

import numpy as np

from .grid import CellSet, dilate, rasterize, rasterize_ring, self_offsets
from .schemas import CGSHOP2027Instance, CGSHOP2027Solution, CutterPath
from .schemas.solution import _DIRECTION_DELTAS

__all__ = ["SolutionValidator", "check_for_errors"]


class SolutionValidator:
    """
    Checks solutions against a fixed instance.

    The instance-derived data -- the rasterized region, the cutter relative to
    its center, and the offsets under which two cutters overlap -- is computed
    once here, so verifying a batch of solutions for the same instance pays for
    it only once.
    """

    def __init__(self, instance: CGSHOP2027Instance) -> None:
        self.instance = instance
        self.region = rasterize(instance.region_to_cover)
        center_x, center_y = instance.cutter_center
        self.cutter = rasterize_ring(instance.cutter).translated(-center_x, -center_y)
        self.collision_offsets = self_offsets(self.cutter)

    def check_for_errors(self, solution: CGSHOP2027Solution) -> list[str]:
        """
        Returns a list of error messages; an empty list means the solution is
        feasible for this instance.
        """
        if solution.instance_uid != self.instance.instance_uid:
            return [
                f"The solution is for instance {solution.instance_uid!r}, "
                f"but was checked against {self.instance.instance_uid!r}."
            ]
        errors = []
        if len(solution.cutters) > self.instance.number_of_cutters:
            errors.append(
                f"The solution uses {len(solution.cutters)} cutters, but only "
                f"{self.instance.number_of_cutters} are available."
            )
        errors.extend(self._check_collisions(solution))
        errors.extend(self._check_coverage(solution))
        return errors

    # -- coverage ----------------------------------------------------------

    def swept_cells(self, solution: CGSHOP2027Solution) -> CellSet:
        """
        The cells the cutters sweep over, clipped to the region's surroundings.
        """
        return dilate(self._anchors(solution), self.cutter)

    def uncovered_cells(self, solution: CGSHOP2027Solution) -> CellSet:
        """
        The cells of the region that no cutter ever sweeps over.
        """
        return self.region.difference(self.swept_cells(solution))

    def _anchors(self, solution: CGSHOP2027Solution) -> CellSet:
        """
        Every anchor position visited by any cutter, clipped to the positions
        from which the cutter can touch the region at all.

        Each step contributes a single slice write, so this costs one operation
        per step rather than one per time unit.
        """
        region_x0, region_y0, region_x1, region_y1 = self.region.bounds
        cutter_x0, cutter_y0, cutter_x1, cutter_y1 = self.cutter.bounds
        origin = (region_x0 - cutter_x1, region_y0 - cutter_y1)
        mask = np.zeros(
            (
                (region_y1 - cutter_y0) - origin[1] + 1,
                (region_x1 - cutter_x0) - origin[0] + 1,
            ),
            dtype=bool,
        )
        for path in solution.cutters:
            _mark_segment(mask, origin, path.start, path.start)
            for segment_start, segment_end in pairwise(path.waypoints()):
                _mark_segment(mask, origin, segment_start, segment_end)
        return CellSet(mask, origin)

    def _check_coverage(self, solution: CGSHOP2027Solution) -> list[str]:
        uncovered = self.uncovered_cells(solution)
        missing = len(uncovered)
        if not missing:
            return []
        witness = next(uncovered.cells())
        return [
            f"The region is not covered: {missing} cells are never swept, "
            f"for example the cell at {witness}."
        ]

    # -- collisions --------------------------------------------------------

    def _check_collisions(self, solution: CGSHOP2027Solution) -> list[str]:
        makespan = max((path.duration for path in solution.cutters), default=0)
        trajectories = [_trajectory(path, makespan) for path in solution.cutters]
        errors = []
        for i, (positions_i, velocities_i) in enumerate(trajectories):
            for j, (positions_j, velocities_j) in enumerate(trajectories[:i]):
                conflict = self._first_conflict(
                    positions_i - positions_j, velocities_i - velocities_j
                )
                if conflict is not None:
                    time, moving = conflict
                    when = (
                        f"while moving between time {time} and {time + 1}"
                        if moving
                        else f"at time {time}"
                    )
                    errors.append(f"Cutters #{j} and #{i} overlap {when}.")
        return errors

    def _first_conflict(
        self, offsets: np.ndarray, relative_velocity: np.ndarray
    ) -> tuple[int, bool] | None:
        """
        The earliest time at which two cutters with the given relative offsets
        overlap, and whether the overlap happens during a move rather than at
        an integer time.

        Overlap at an integer time is a lookup of the offset. In between, the
        offset travels along the segment `offset + s * velocity`; since each
        cutter moves by at most one unit per axis, the relative velocity is one
        of only four kinds, and the lattice points that segment can pass
        through are:

        | relative velocity   | additionally reachable offsets       |
        | ------------------- | ------------------------------------ |
        | `0`, `(+-1, 0)`, `(0, +-1)` | none, the endpoints cover it |
        | `(+-2, 0)`, `(0, +-2)`      | the midpoint (a head-on swap)|
        | `(+-1, +-1)`                | the two corners of the square|
        """
        candidates: list[tuple[int, bool]] = []

        overlapping = self.collision_offsets.contains_points(offsets)
        if overlapping.any():
            candidates.append((int(np.argmax(overlapping)), False))

        # Head-on: the two cutters close along the same axis and pass through
        # the offset halfway between, which no integer time ever samples.
        head_on = np.flatnonzero(np.abs(relative_velocity).max(axis=1) == 2)
        if head_on.size:
            midpoints = offsets[head_on] + relative_velocity[head_on] // 2
            hits = self.collision_offsets.contains_points(midpoints)
            if hits.any():
                candidates.append((int(head_on[np.argmax(hits)]), True))

        # Diagonal: the offset moves along the diagonal of a unit square, whose
        # interior meets the cells of the two off-diagonal corners.
        diagonal = np.flatnonzero(
            (relative_velocity[:, 0] != 0) & (relative_velocity[:, 1] != 0)
        )
        if diagonal.size:
            for axis in (0, 1):
                corner = offsets[diagonal].copy()
                corner[:, axis] += relative_velocity[diagonal, axis]
                hits = self.collision_offsets.contains_points(corner)
                if hits.any():
                    candidates.append((int(diagonal[np.argmax(hits)]), True))

        if not candidates:
            return None
        # An overlap at time t precedes one during the move starting at t.
        return min(candidates, key=lambda entry: (entry[0], entry[1]))


def check_for_errors(
    instance: CGSHOP2027Instance, solution: CGSHOP2027Solution
) -> list[str]:
    """
    Verifies the given solution against the provided instance and returns a
    list of error messages if any issues are found.

    Use `SolutionValidator` directly when checking several solutions for the
    same instance, to reuse the instance-derived precomputation.
    """
    return SolutionValidator(instance).check_for_errors(solution)


def _trajectory(path: CutterPath, makespan: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Expands a compressed trajectory into the position at every integer time and
    the velocity during every unit time interval.

    A cutter that finishes early stays parked where it is, which -- since every
    cutter returns to its start -- is its start position.
    """
    velocity = np.zeros((makespan, 2), dtype=np.int64)
    if path.steps:
        deltas = np.array(
            [_DIRECTION_DELTAS[direction] for direction, _ in path.steps],
            dtype=np.int64,
        )
        counts = np.array([units for _, units in path.steps], dtype=np.int64)
        velocity[: path.duration] = np.repeat(deltas, counts, axis=0)
    position = np.empty((makespan + 1, 2), dtype=np.int64)
    position[0] = path.start
    np.cumsum(velocity, axis=0, out=position[1:])
    position[1:] += np.asarray(path.start, dtype=np.int64)
    return position, velocity


def _mark_segment(
    mask: np.ndarray,
    origin: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """
    Marks the lattice points of an axis-parallel segment, clipping to the mask.
    """
    (x0, y0), (x1, y1) = start, end
    origin_x, origin_y = origin
    height, width = mask.shape
    if y0 == y1:
        row = y0 - origin_y
        if not 0 <= row < height:
            return
        low, high = sorted((x0, x1))
        low, high = max(low - origin_x, 0), min(high - origin_x, width - 1)
        if low <= high:
            mask[row, low : high + 1] = True
    else:
        column = x0 - origin_x
        if not 0 <= column < width:
            return
        low, high = sorted((y0, y1))
        low, high = max(low - origin_y, 0), min(high - origin_y, height - 1)
        if low <= high:
            mask[low : high + 1, column] = True
