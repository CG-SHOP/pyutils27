import os
import sys
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

from ..schemas.instance import CGSHOP2027Instance
from .instance_base_database import InstanceBaseDatabase

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class InstanceFileDatabase(InstanceBaseDatabase):
    """
    This class allows to easily read instances from a folder if the instance files
    follow the naming convention 'instance-name.instance.json'. It allows subfolder
    but no symbolic links.

    Note that every `.json` file is considered an instance file, and the instance
    name is everything before the first dot; a folder that also holds solutions
    will try to read `NAME.solution.json` as an instance.
    """

    def __init__(self, path: str | PathLike[str], enable_cache: bool = False):
        """
        Create an InstanceDatabase that searches in a specified folder for instances.
        :param path: Path to the folder that contains the instance files (e.g. the folder
                        that contains the extracted zips). The instance files can be
                        in subfolders but have the names have to be NAME.instance.json.
        :param enable_cache: Should the loaded instances be cached? This can take quite
                        a lot of memory
        """
        super().__init__(path, enable_cache)

    def _iterate_paths(self) -> Iterator[Path]:
        for root, dirs, files in os.walk(self._path, topdown=True):
            dirs[:] = [d for d in dirs if not self._is_hidden_folder(d)]
            for file in files:
                if self._filename_fits_instance_convention(
                    file
                ) and not self._is_hidden_folder_name(file):
                    yield Path(root) / file

    def _find_path(self, name: str) -> Path:
        for instance_path in self._iterate_paths():
            if self._filename_fits_name(instance_path.name, name):
                return instance_path
        msg = f"Did not find a suitable file for {name} in {self._path}"
        raise KeyError(msg)

    @override
    def _extract_instance_name_from_path(self, path: Path) -> str:
        return path.name.split(".")[0]

    @override
    def __iter__(self) -> Iterator[CGSHOP2027Instance]:
        """
        Iterate over all instance files.
        :return: Instance objects
        """
        for instance_path in self._iterate_paths():
            instance_name = self._extract_instance_name_from_path(instance_path)
            if instance_name in self._cache:
                yield self._cache[instance_name]
            else:
                yield self._cache_and_return(self.read(instance_path))

    @override
    def __getitem__(self, name: str) -> CGSHOP2027Instance:
        """
        Returns the instance of a specific name or throws an KeyError.
        :param name: Name of the instance.
        :return:
        """
        if name in self._cache:
            return self._cache[name]
        path = self._find_path(name)
        return self._cache_and_return(self.read(path))
