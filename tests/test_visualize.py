from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution
from cgshop2027_pyutils.verify import SolutionValidator
from cgshop2027_pyutils.visualize import (
    create_instance_plot,
    create_solution_animation,
    create_solution_plot,
)

UNIT_CUTTER = {"x": [0, 1, 1, 0], "y": [0, 0, 1, 1]}


def make_instance(size: int = 4, holes=(), cutter=UNIT_CUTTER, center=(0, 0)):
    return CGSHOP2027Instance(
        instance_uid="demo",
        region_to_cover={
            "outer_boundary": {"x": [0, size, size, 0], "y": [0, 0, size, size]},
            "inner_boundaries": list(holes),
        },
        cutter=cutter,
        cutter_center=center,
        number_of_cutters=4,
    )


def boustrophedon(y0: int = 0, rows: int = 4, width: int = 3):
    """A closed tour sweeping `rows` rows with a unit cutter."""
    steps, left = [], True
    for _ in range(rows - 1):
        steps.append(("E" if left else "W", width))
        left = not left
        steps.append(("N", 1))
    steps.append(("E" if left else "W", width))
    if not left:
        left = not left
    steps.append(("S", rows - 1))
    if not left:
        steps.append(("W", width))
    return {"start": (0, y0), "steps": steps}


FULL_SWEEP = boustrophedon()
HOLE = {"x": [1, 1, 3, 3], "y": [1, 3, 3, 1]}


def make_solution(*paths):
    return CGSHOP2027Solution(instance_uid="demo", cutters=list(paths) or [FULL_SWEEP])


def renders(figure: Figure) -> bool:
    """Draws the figure to prove the artists survive a real render pass."""
    figure.savefig(BytesIO(), format="png", dpi=50)
    return True


# --------------------------------------------------------------------------
# instance plot
# --------------------------------------------------------------------------


def test_instance_plot_has_a_panel_for_the_region_and_the_cutter():
    figure = create_instance_plot(make_instance())
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2


def test_instance_plot_renders():
    assert renders(create_instance_plot(make_instance(holes=[HOLE])))


def test_instance_plot_covers_the_whole_region():
    figure = create_instance_plot(make_instance(size=7))
    region_axes = figure.axes[0]
    x_min, x_max = region_axes.get_xlim()
    y_min, y_max = region_axes.get_ylim()
    assert x_min <= 0
    assert x_max >= 7
    assert y_min <= 0
    assert y_max >= 7


def test_instance_plot_reports_the_cell_counts():
    figure = create_instance_plot(make_instance(size=4, holes=[HOLE]))
    assert "12 cells" in figure.axes[0].get_title()  # 16 minus a 2x2 hole
    assert "demo" in figure.get_suptitle()


# --------------------------------------------------------------------------
# solution plot
# --------------------------------------------------------------------------


def test_solution_plot_renders():
    assert renders(create_solution_plot(make_instance(), make_solution()))


def test_solution_plot_reports_a_feasible_solution():
    figure = create_solution_plot(make_instance(), make_solution())
    assert "feasible" in figure.get_suptitle()
    assert "0 cells uncovered" in figure.axes[0].get_title()


def test_solution_plot_reports_the_gaps_of_an_incomplete_solution():
    figure = create_solution_plot(make_instance(), make_solution(boustrophedon(rows=2)))
    assert "1 error" in figure.get_suptitle()
    assert "8 cells uncovered" in figure.axes[0].get_title()


def test_solution_plot_renders_holes_and_several_cutters():
    solution = make_solution(boustrophedon(rows=2), boustrophedon(y0=2, rows=2))
    assert renders(create_solution_plot(make_instance(holes=[HOLE]), solution))


def test_solution_plot_includes_area_swept_outside_the_region():
    """The view has to grow to fit a cutter that wanders off."""
    detour = {"start": (0, 0), "steps": [("W", 3), ("E", 3)]}
    figure = create_solution_plot(make_instance(), make_solution(detour))
    x_min, _ = figure.axes[0].get_xlim()
    assert x_min <= -3


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------


def frames_of(animation: FuncAnimation) -> list[int]:
    return list(animation.new_frame_seq())


def covered_cells(image) -> int:
    """
    The number of painted cells of a coverage raster.

    Goes through `getmaskarray` on purpose: once every cell is covered numpy
    collapses the all-False mask to the `nomask` scalar, and `~image.mask`
    would then count 1 instead of the whole grid.
    """
    return int((~np.ma.getmaskarray(image.get_array())).sum())


def test_animation_shows_every_time_step_of_a_short_trajectory():
    solution = make_solution()
    animation = create_solution_animation(make_instance(), solution)
    assert isinstance(animation, FuncAnimation)
    assert len(frames_of(animation)) == solution.cutters[0].duration + 1


def test_an_explicit_step_is_honoured():
    animation = create_solution_animation(make_instance(), make_solution(), step=4)
    duration = make_solution().cutters[0].duration
    assert len(frames_of(animation)) == len(range(0, duration + 1, 4)) + 1


def test_the_frame_budget_is_respected():
    long_tour = {"start": (0, 0), "steps": [("E", 200), ("W", 200)]}
    animation = create_solution_animation(
        make_instance(), make_solution(long_tour), max_frames=10
    )
    # The final time is always shown, which may cost one frame over budget.
    assert len(frames_of(animation)) <= 11


def test_the_last_frame_is_the_final_time():
    long_tour = {"start": (0, 0), "steps": [("E", 7), ("W", 7)]}
    solution = make_solution(long_tour)
    animation = create_solution_animation(make_instance(), solution, max_frames=4)
    for frame in frames_of(animation):
        animation._func(frame)
    assert "time 14 / 14" in animation._fig.axes[0].get_title()


def test_a_stationary_solution_animates_a_single_frame():
    animation = create_solution_animation(
        make_instance(), make_solution({"start": (0, 0), "steps": []})
    )
    assert len(frames_of(animation)) == 1


@pytest.mark.parametrize("max_frames", [300, 5, 2])
def test_coverage_grows_monotonically_to_the_full_swept_area(max_frames: int):
    """
    Striding must not lose the placements between two frames: whatever the
    frame budget, the animation has to end on exactly the area the verifier
    reports as swept.
    """
    instance, solution = make_instance(), make_solution()
    expected = len(SolutionValidator(instance).swept_cells(solution))
    animation = create_solution_animation(instance, solution, max_frames=max_frames)
    covered = []
    for frame in frames_of(animation):
        (image, *_) = animation._func(frame)
        covered.append(covered_cells(image))
    assert covered == sorted(covered)
    assert covered[-1] == expected
    assert covered[0] < covered[-1]


def test_animation_restarts_cleanly_when_it_loops():
    """Replaying from the first frame must not keep the coverage of the last."""
    instance, solution = make_instance(), make_solution()
    animation = create_solution_animation(instance, solution)
    frames = frames_of(animation)
    for frame in frames:
        animation._func(frame)
    (image, *_) = animation._func(frames[0])
    first_again = covered_cells(image)
    assert first_again == len(SolutionValidator(instance).cutter)


def test_animation_saves_as_a_gif(tmp_path: Path):
    path = tmp_path / "tour.gif"
    animation = create_solution_animation(
        make_instance(), make_solution(), max_frames=4
    )
    animation.save(path, writer="pillow", dpi=40)
    assert path.stat().st_size > 0
