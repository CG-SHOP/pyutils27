from os import PathLike
from pathlib import Path
from types import TracebackType
from zipfile import ZipFile

from ..schemas.instance import CGSHOP2027Instance
from ..schemas.solution import CGSHOP2027Solution


class ZipWriter:
    """
    Writes instances and solutions into a ZIP archive, one JSON file each,
    following the `NAME.instance.json` / `NAME.solution.json` convention.

    Meant to be used as a context manager:
    ```
    with ZipWriter("solutions.zip") as writer:
        writer.add_solution(solution)
    ```
    """

    def __init__(self, path: str | PathLike[str]):
        self._path: Path = Path(path)
        if self._path.exists():
            msg = f"File {self._path} already exists."
            raise FileExistsError(msg)
        self._zip: ZipFile = ZipFile(self._path, "w")

    def add_instance(self, instance: CGSHOP2027Instance):
        self._zip.writestr(
            f"{instance.instance_uid}.instance.json", instance.model_dump_json()
        )

    def add_solution(self, solution: CGSHOP2027Solution):
        self._zip.writestr(
            f"{solution.instance_uid}.solution.json", solution.model_dump_json()
        )

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self.close()
        return False
