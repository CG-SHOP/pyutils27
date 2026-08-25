# pyutils27 (cgshop2027-pyutils)

Utilities for working with CG:SHOP 2027 instances & solutions. This is a pure
Python package containing a parser/verifier for our instance format, solution
format and a verifier for solutions.

---

## Installation

```bash
pip install cgshop2027-pyutils
```

The current state of `main`, ahead of the latest release:

```bash
pip install git+https://github.com/CG-SHOP/pyutils27
```

From a local clone (editable, for development):

```bash
git clone https://github.com/CG-SHOP/pyutils27
cd pyutils27
pip install -e ".[dev]"
```

---

## Layout

```
src/cgshop2027_pyutils/
├── schemas/            # Pydantic models for instances & solutions
├── io/                 # reading instances & solutions from paths or file objects
├── zip/                # safe reading/writing of instance & solution archives
├── instance_database/  # query instances from a folder or a zip
├── grid.py             # cell-set algebra on the integer grid
├── verify.py           # checking a solution against its instance
└── visualize.py        # matplotlib plots and animations
tests/                  # pytest test suite
```

---

## Usage

Verifying a solution against its instance:

```python
from cgshop2027_pyutils.io import read_instance, read_solution
from cgshop2027_pyutils.verify import check_for_errors

errors = check_for_errors(
    read_instance("demo.instance.json"),
    read_solution("demo.solution.json"),
)
```

`check_for_errors` returns a list of human-readable messages; an empty list
means the solution is feasible. When checking many solutions for the same
instance, use `SolutionValidator(instance)` directly so that the
instance-derived precomputation is only paid once.

A solution consists of exactly `number_of_cutters` tours, one per cutter:

```python
solution = CGSHOP2027Solution(
    instance_uid="demo",
    tours=[{"x": [0, 3, 3, 0], "y": [0, 0, 1, 1]}, {"x": [7], "y": [7]}],
)
```

Each tour lists the points the cutter's _center_ passes through and is closed
implicitly, so the cutter travels from the last point back to the first one.
Consecutive points must be axis-parallel; a point may be visited again later,
but not immediately again. A tour of a single point is a cutter that stays where
it is. Cutters do not interfere with each other and may leave the region; all
that is required is that together they sweep the whole region.

The objective is `solution.max_tour_length`, the length of the longest tour, and
it is to be minimized.

Bundling instances or solutions into an archive:

```python
from cgshop2027_pyutils.zip import ZipWriter

with ZipWriter("solutions.zip") as writer:
    for solution in solutions:
        writer.add_solution(solution)
```

Reading a submitted archive, with size, CRC and path-traversal checks:

```python
from cgshop2027_pyutils.zip import BadSolutionFile, ZipSolutionIterator, ZipReaderError

try:
    for solution in ZipSolutionIterator("solutions.zip"):
        print(solution.instance_uid)
except (ZipReaderError, BadSolutionFile) as e:
    print(f"Rejected: {e}")
```

`example_zip.ipynb` walks the whole submission path end to end: reading the
instances, checking solutions against them, packing an archive, and what a
rejection looks like. It runs on the two archives next to it:
`example_instances.zip`, a copy of the published example instances, and
`example_solutions.zip`, one feasible solution for each of them to check against
and to beat.

Looking instances up by name, from either a folder or a zip:

```python
from cgshop2027_pyutils.instance_database import InstanceDatabase

db = InstanceDatabase("instances.zip")
instance = db["demo"]
for instance in db:
    ...
```

Plotting an instance, a solution, or an animation of the cutters at work:

```python
from cgshop2027_pyutils.visualize import (
    create_instance_plot,
    create_solution_animation,
    create_solution_plot,
)

create_instance_plot(instance).savefig("instance.png")
create_solution_plot(instance, solution).savefig("solution.png")

animation = create_solution_animation(instance, solution)
animation.save("solution.gif", writer="pillow")
```

The solution plot shades the swept area, marks any cells left uncovered in red
and draws one tour per cutter. The animation strides through the tours to fit
`max_frames` (300 by default), filling in coverage as the cutters move; pass
`step` to control the stride yourself.

All three take `bare=True`, which drops the axes, the grid and every title and
trims the figure to the drawing, for a picture meant for a paper or a slide
rather than for debugging:

```python
create_solution_plot(instance, solution, bare=True).savefig("solution.pdf")
```

Instance files must be named `NAME.instance.json`. A lookup key may be the bare
name, the file name, or a path to it: `db["demo"]`, `db["demo.instance.json"]`
and `db["batch1/demo.instance.json"]` all resolve to the same instance. Note
that every `.json` file in the folder or archive is treated as an instance, so
keep solutions in a separate location.

---

## Development

Set up the tooling (linting & formatting run through pre-commit):

```bash
pip install -e ".[dev]"
pre-commit install
```

Run the checks manually:

```bash
ruff check .
ruff format .
pytest
```

Linting, formatting and import sorting are all handled by
[ruff](https://docs.astral.sh/ruff/); its configuration lives in
`pyproject.toml`.

---

## Contributing

Issues & PRs welcome. Please:

1. Add/adjust unit tests for new behavior
2. Keep the public API documented here
3. Run the test suite before submitting

---

## Changelog

- **0.1.0** (2026-08-25): First release. Instance and solution schemas, the
  solution verifier, archive reading and writing, the instance database, and
  plotting and animation.

---

## License

MIT, see `LICENSE`.
