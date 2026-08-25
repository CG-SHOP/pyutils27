"""
Verification of a solution against its instance.

The checks that need only the solution itself (well-formed, closed, orthogonal
tours) live in the schema. What is left needs both objects: the solution must
provide one tour per available cutter, and the tours together must sweep the
whole region.

The coverage check avoids walking the tours cell by cell: the swept area is
`cutter (+) anchors`, the Minkowski sum of the cutter with the set of all
anchor positions any cutter ever visits. Unioning the anchors first makes
revisits free, and one dilation replaces one placement per cutter per unit of
travel. Continuous motion adds nothing to sweep, because for a union of unit
cells `C` and a unit vector `e` we have `C (+) [p, p+e] = (C+p) | (C+p+e)`.
"""

import numpy as np

from .grid import CellSet, dilate, rasterize, rasterize_ring
from .schemas import CGSHOP2027Instance, CGSHOP2027Solution

__all__ = ["SolutionValidator", "check_for_errors"]


class SolutionValidator:
    """
    Checks solutions against a fixed instance.

    The instance-derived data -- the rasterized region and the cutter relative
    to its center -- is computed once here, so verifying a batch of solutions
    for the same instance pays for it only once.
    """

    def __init__(self, instance: CGSHOP2027Instance) -> None:
        self.instance = instance
        self.region = rasterize(instance.region_to_cover)
        center_x, center_y = instance.cutter_center
        self.cutter = rasterize_ring(instance.cutter).translated(-center_x, -center_y)

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
        given = len(solution.tours)
        if given != self.instance.number_of_cutters:
            errors.append(
                f"The solution has {given} tour{'' if given == 1 else 's'}, but "
                f"the instance calls for exactly "
                f"{self.instance.number_of_cutters}."
            )
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

        Each edge contributes a single slice write, so this costs one operation
        per edge rather than one per unit of travel.
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
        for tour in solution.tours:
            # A tour of a single point has no edges to walk, but its cutter
            # still sweeps where it stands.
            _mark_segment(mask, origin, tour.start, tour.start)
            for edge_start, edge_end in tour.edges():
                _mark_segment(mask, origin, edge_start, edge_end)
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
