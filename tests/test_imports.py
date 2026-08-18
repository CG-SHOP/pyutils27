"""
Test that all modules can be imported successfully.
This catches compatibility issues with typing features across Python versions.
"""

import sys


def test_python_version():
    """Verify we're running on a supported Python version."""
    assert sys.version_info >= (3, 10), "Python 3.10+ required"


def test_import_package():
    """The top-level package imports and exposes a version."""
    import cgshop2027_pyutils

    assert cgshop2027_pyutils.__version__


def test_import_schemas():
    """Test that schema classes can be imported."""
    from cgshop2027_pyutils.schemas import CGSHOP2027Instance, CGSHOP2027Solution

    assert CGSHOP2027Instance is not None
    assert CGSHOP2027Solution is not None


def test_import_io():
    """Test that IO functions can be imported."""
    from cgshop2027_pyutils.io import read_instance, read_solution

    assert read_instance is not None
    assert read_solution is not None


def test_import_verify():
    """Test that the verifier can be imported."""
    from cgshop2027_pyutils.verify import SolutionValidator, check_for_errors

    assert SolutionValidator is not None
    assert check_for_errors is not None


def test_import_zip():
    """Test that the zip helpers can be imported."""
    from cgshop2027_pyutils.zip import (
        BadSolutionFile,
        ZipReaderError,
        ZipSolutionIterator,
        ZipWriter,
    )

    assert ZipSolutionIterator is not None
    assert ZipWriter is not None
    assert ZipReaderError is not None
    assert BadSolutionFile is not None


def test_import_instance_database():
    """Test that the instance database can be imported."""
    from cgshop2027_pyutils.instance_database import InstanceDatabase

    assert InstanceDatabase is not None
