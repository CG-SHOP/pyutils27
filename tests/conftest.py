import matplotlib
import pytest

# Must happen before pyplot is imported anywhere: the interactive default would
# try to open windows during the test run.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_figures():
    """Keeps the suite below matplotlib's open-figure warning threshold."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
