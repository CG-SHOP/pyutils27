import sys
from collections.abc import Iterator
from os import PathLike
from pathlib import Path
from zipfile import ZipFile, ZipInfo

from ..schemas.instance import CGSHOP2027Instance
from .instance_base_database import InstanceBaseDatabase

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class InstanceZipDatabase(InstanceBaseDatabase):
    """
    This class allows to easily read instances from a zipfile if the instance files
    follow the naming convention 'instance-name.instance.json'. It allows subfolder
    but no symbolic links.

    Note that every `.json` file is considered an instance file, and the instance
    name is everything before the first dot; an archive that also holds solutions
    will try to read `NAME.solution.json` as an instance.
    """

    def __init__(self, path: str | PathLike[str], enable_cache: bool = False):
        """
        Create an InstanceDatabase that searches in a specified zipfile for instances.
        :param path: Path to the zipfile that contains the instance files.
                        The instance files can be in subfolders but have the names have to be
                        NAME.instance.json.
        :param enable_cache: Should the loaded instances be cached? This can take quite
                        a lot of memory
        """
        super().__init__(path, enable_cache)
        self._zipfile: ZipFile = ZipFile(self._path)

    def _find_path(self, name: str) -> ZipInfo:
        for instance_path in self._zipfile.filelist:
            if self._filename_fits_name(Path(instance_path.filename).name, name):
                return instance_path
        msg = f"Did not find a suitable file for {name} in {self._path}"
        raise KeyError(msg)

    @override
    def __iter__(self) -> Iterator[CGSHOP2027Instance]:
        """
        Iterate over all instance files.
        :return: Instance objects
        """
        for file_data in self._zipfile.filelist:
            if self._filename_fits_instance_convention(
                file_data.filename
            ) and not self._is_hidden_folder(file_data.filename):
                instance_name = self._extract_instance_name_from_path(
                    Path(file_data.filename)
                )
                try:
                    yield self._cache[instance_name]
                except KeyError:
                    yield self._cache_and_return(
                        self.read(self._zipfile.open(file_data.filename))
                    )

    @override
    def __getitem__(self, name: str) -> CGSHOP2027Instance:
        """
        Returns the instance of a specific name or throws an KeyError.
        :param name: Name of the instance.
        :return:
        """
        try:
            return self._cache[name]
        except KeyError:
            path = self._find_path(name)
            return self._cache_and_return(self.read(self._zipfile.open(path)))
