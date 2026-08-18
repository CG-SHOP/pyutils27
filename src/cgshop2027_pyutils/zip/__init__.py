"""
Safe reading and writing of instance/solution archives.
"""

from .zip_processor import BadSolutionFile, ZipSolutionIterator
from .zip_reader_errors import ZipReaderError
from .zip_writer import ZipWriter

__all__ = ["BadSolutionFile", "ZipReaderError", "ZipSolutionIterator", "ZipWriter"]
