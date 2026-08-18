import abc
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from typing import Literal

from ..io import FileLike, read_instance
from ..schemas.instance import CGSHOP2027Instance


class InstanceBaseDatabase(abc.ABC):
    """
    Abstract base class (ABC) for accessing instance databases.
    Subclasses must implement methods to iterate over instances and retrieve a
    specific instance by name.
    """

    def __init__(self, path: str | PathLike[str], enable_cache: bool = False):
        """
        Initializes the InstanceBaseDatabase with a path and optional caching.
        :param path: Path to the folder containing the instance files. The instance files
                     can be in subfolders, but their names must follow the pattern
                     NAME.instance.json.
        :param enable_cache: Whether to enable caching of loaded instances. Caching can
                             consume a significant amount of memory.
        """
        self._path: Path = Path(path)
        self._is_cache_enabled: bool = enable_cache
        self._cache: dict[str, CGSHOP2027Instance] = {}
        self.extension: Literal[".json"] = ".json"

        if not self._path.exists():
            msg = f"The folder {self._path.resolve()} does not exist"
            raise ValueError(msg)

    def _filename_fits_name(self, filename: str, name: str) -> bool:
        """Checks if the filename matches the given instance name and follows the naming convention."""
        if not self._filename_fits_instance_convention(filename):
            return False
        return filename.split(".", maxsplit=1)[0] == name

    def _filename_fits_instance_convention(self, filename: str) -> bool:
        """Checks if the file follows the required instance file naming convention (i.e., ends with .json)."""
        return filename.endswith(self.extension)

    def read(self, f: FileLike) -> CGSHOP2027Instance:
        """Reads an instance from a file."""
        return read_instance(f)

    def _cache_and_return(self, instance: CGSHOP2027Instance) -> CGSHOP2027Instance:
        """Caches the instance if caching is enabled and returns the instance."""
        if self._is_cache_enabled:
            self._cache[instance.instance_uid] = instance
        return instance

    def _is_hidden_folder_name(self, name: str) -> bool:
        """
        Checks if the folder is hidden based on UNIX naming conventions.
        Hidden folders start with a dot (.) or __ (Mac OS).
        """
        return name.startswith((".", "__"))

    def _is_hidden_folder(self, path: str) -> bool:
        """Checks if any part of the folder path is hidden."""
        return any(self._is_hidden_folder_name(folder) for folder in Path(path).parts)

    def _extract_instance_name_from_path(self, path: Path) -> str:
        """Extracts the instance name from the given file path."""
        filename = path.name
        return filename.split(".")[0]

    @abc.abstractmethod
    def __iter__(self) -> Iterator[CGSHOP2027Instance]:
        """
        Abstract method to iterate over all instances in the database.
        :return: An iterable of CGSHOP2027Instance objects.
        """

    @abc.abstractmethod
    def __getitem__(self, name: str) -> CGSHOP2027Instance:
        """
        Abstract method to retrieve an instance by name.
        :param name: The name of the instance to retrieve.
        :return: The CGSHOP2027Instance object corresponding to the given name.
        :raises KeyError: If the instance with the given name is not found.
        """
