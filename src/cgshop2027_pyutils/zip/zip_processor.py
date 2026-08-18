"""
This file contains the ZipSolutionIterator which can read a zip and return all
solutions in it. It is designed to be robust and include basic security
features.
"""

import sys
from collections.abc import Iterator, Sequence
from os import PathLike
from typing import Any, BinaryIO
from zipfile import BadZipFile, ZipFile

from pydantic import ValidationError

from ..io import read_solution
from ..schemas import CGSHOP2027Solution
from .zip_reader_errors import (
    BadZipChecker,
    InvalidZipError,
    NoSolutionsError,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class BadSolutionFile(Exception):
    """Custom exception raised for invalid solution files in the zip."""

    def __init__(self, msg: str, file_name: str):
        super().__init__(msg)
        self.file_name: str = file_name

    @override
    def __str__(self) -> str:
        return self.args[0]


class ZipSolutionIterator:
    """
    Iterates over all solutions in a zip file.
    After initializing the class, use the instance as an iterator.
    Example:
    ```
    zsi = ZipSolutionIterator("./myzip.zip")
    for solution in zsi:
        print(solution.instance_uid)
    ```
    """

    def __init__(
        self,
        path_or_file: BinaryIO | str | PathLike[str],
        file_size_limit: int = 250 * 1_000_000,  # 250 MB file size limit
        zip_size_limit: int = 2_000 * 1_000_000,  # 2 GB zip size limit
        solution_extensions: Sequence[str] = (".solution.json", ".sol.json"),
    ):
        self.path: BinaryIO | str | PathLike[str] = path_or_file
        self._checker: BadZipChecker = BadZipChecker(
            file_size_limit=file_size_limit, zip_size_limit=zip_size_limit
        )
        self._solution_extensions: Sequence[str] = list(solution_extensions)

    def _check_if_bad_zip(self, zip_file: ZipFile):
        """Checks the validity and security of the zip file using the BadZipChecker."""
        self._checker(zip_file)

    def _is_hidden_folder_name(self, name: str) -> bool:
        """Returns True if the folder is hidden (starts with . or __)."""
        return len(name) > 1 and (name[0] == "." or name[:2] == "__")

    def _is_solution_filename(self, name: str) -> bool:
        """Checks if the file is a valid solution file based on its extension and visibility."""
        # Check extension and skip hidden files/directories
        return any(
            name.lower().endswith(extension) for extension in self._solution_extensions
        ) and not any(self._is_hidden_folder_name(s) for s in name.split("/"))

    def _iterate_solution_filenames(self, zip_file: ZipFile) -> Iterator[str]:
        """Yields filenames that match the allowed solution extensions."""
        had_filename = False
        for filename in zip_file.namelist():
            if self._is_solution_filename(filename):
                had_filename = True
                yield filename
        if not had_filename:
            raise NoSolutionsError

    def __iter__(self) -> Iterator[CGSHOP2027Solution]:
        """
        Iterates over all solutions in the zip file.
        :return: An iterator over valid CGSHOP2027Solution objects.
        """
        found_a_solution = False
        try:
            with ZipFile(self.path) as zip_file:
                self._check_if_bad_zip(zip_file)
                for file_name in self._iterate_solution_filenames(zip_file):
                    meta: dict[str, Any] = {
                        "zip_info": {
                            "zip_file": zip_file.filename,
                            "file_in_zip": file_name,
                        }
                    }
                    with zip_file.open(file_name, "r") as sol_file:
                        try:
                            solution = read_solution(sol_file)
                        except ValidationError as e:
                            msg = f"Error in file '{file_name}': {e}"
                            raise BadSolutionFile(msg, file_name=str(file_name)) from e
                        solution.meta.update(meta)
                        found_a_solution = True
                        yield solution
        except BadZipFile as e:
            msg = f"Invalid ZIP file: {e}"
            raise InvalidZipError(msg) from e

        if not found_a_solution:
            raise NoSolutionsError
