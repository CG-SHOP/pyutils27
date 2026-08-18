import sys
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveInt, model_validator

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from .instance import Coordinate

Direction = Literal["N", "E", "S", "W", "I"]
Step = tuple[Direction, PositiveInt]

# `y` grows northwards, matching the orientation convention of
# `PointSequence.is_ccw`. "I" is idling, which costs time but no movement.
_DIRECTION_DELTAS: dict[Direction, tuple[Coordinate, Coordinate]] = {
    "N": (0, 1),
    "E": (1, 0),
    "S": (0, -1),
    "W": (-1, 0),
    "I": (0, 0),
}


class CutterPath(BaseModel):
    """
    This schema represents the trajectory of a single cutter, given as a
    start position followed by a sequence of axis-parallel moves and idle
    periods.

    Each step is a `[direction, units]` pair; one unit of travel and one unit
    of idling each take one unit of time.
    """

    start: tuple[Coordinate, Coordinate] = Field(
        ...,
        description="Position at which the cutter's center is placed at time 0.",
    )

    steps: list[Step] = Field(
        ...,
        description="Steps of the trajectory, each a pair of a direction "
        "('N', 'E', 'S', 'W', or 'I' for idling) and a positive number of units.",
    )

    @property
    def duration(self) -> int:
        """
        The number of time units the trajectory takes, idling included.
        """
        return sum(units for _, units in self.steps)

    @property
    def end(self) -> tuple[Coordinate, Coordinate]:
        """
        The position of the cutter's center after the last step.
        """
        x, y = self.start
        for direction, units in self.steps:
            dx, dy = _DIRECTION_DELTAS[direction]
            x += dx * units
            y += dy * units
        return (x, y)

    def waypoints(self) -> Iterator[tuple[Coordinate, Coordinate]]:
        """
        Yields the position of the cutter's center at the start and after each
        step; hence `len(steps) + 1` positions.

        This is the trajectory's corner points, which is what a drawing or a
        segment-wise sweep needs; use `positions()` when a value per unit of
        time is required.
        """
        x, y = self.start
        yield (x, y)
        for direction, units in self.steps:
            dx, dy = _DIRECTION_DELTAS[direction]
            x += dx * units
            y += dy * units
            yield (x, y)

    def positions(self) -> Iterator[tuple[Coordinate, Coordinate]]:
        """
        Yields the position of the cutter's center at every time step,
        starting with `start` at time 0; hence `duration + 1` positions.
        """
        x, y = self.start
        yield (x, y)
        for direction, units in self.steps:
            dx, dy = _DIRECTION_DELTAS[direction]
            for _ in range(units):
                x += dx
                y += dy
                yield (x, y)


class CGSHOP2027Solution(BaseModel):
    """
    This schema represents a solution for the CG:SHOP 2027 challenge.

    A solution consists of one trajectory per available cutter. Whether these
    trajectories actually match the instance -- the right number of cutters,
    staying within the region, and covering it -- cannot be decided from the
    solution alone and is checked against the instance separately.
    """

    content_type: Literal["CGSHOP2027_Solution"] = "CGSHOP2027_Solution"

    instance_uid: str = Field(..., description="Unique identifier of the instance.")

    cutters: list[CutterPath] = Field(
        ..., description="The trajectory of each cutter used by this solution."
    )

    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata about the solution, e.g., the name of the algorithm used to compute it.",
    )

    @model_validator(mode="after")
    def _check_each_cutter_returns_to_start(self) -> Self:
        for i, cutter_path in enumerate(self.cutters):
            s, e = cutter_path.start, cutter_path.end
            if s != e:
                raise ValueError(
                    f"Cutter #{i} does not return to its origin ({s[0]},{s[1]}) "
                    + f"but ends up at ({e[0]}, {e[1]})"
                )
        return self
