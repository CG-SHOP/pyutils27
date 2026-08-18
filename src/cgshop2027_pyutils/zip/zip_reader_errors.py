from pathlib import Path
from zipfile import ZipFile


class ZipReaderError(Exception):
    pass


class InvalidFileName(ZipReaderError):
    def __init__(self, file_name: str):
        self.file_name: str = file_name
        super().__init__(
            f"The ZIP archive contains the invalid file name: '{file_name}'!"
        )


class FileTooLargeError(ZipReaderError):
    def __init__(self, file_name: str, file_size: int, file_size_limit: int):
        self.file_name: str = file_name
        self.file_size: int = file_size
        self.file_size_limit: int = file_size_limit
        super().__init__(
            f"The ZIP archive contains the file '{self.file_name}' with a size "
            f"of {self.file_size / 1_000_000} MB (only {self.file_size_limit / 1_000_000} MB allowed)!"
        )


class ZipTooLargeError(ZipReaderError):
    def __init__(self, decompressed_size: int, decompressed_size_limit: int):
        self.decompressed_size: int = decompressed_size
        self.decompressed_size_limit: int = decompressed_size_limit
        super().__init__(
            f"The ZIP archive has a total decompressed size of {self.decompressed_size / 1_000_000} MB "
            f"(only {self.decompressed_size_limit / 1_000_000} MB allowed)!"
        )


class NoSolutionsError(ZipReaderError):
    def __init__(self):
        super().__init__(
            "The ZIP archive does not appear to contain any solution! "
            "Make sure, your solutions have the extension '.solution.json'."
        )


class InvalidZipError(ZipReaderError):
    def __init__(self, message: str):
        super().__init__(
            f"The ZIP archive is corrupted and could not be decompressed: {message}!"
        )


class BadZipChecker:
    """
    Check if zip is bad/malicious/corrupted.
    """

    def __init__(self, file_size_limit: int, zip_size_limit: int):
        self.file_size_limit: int = file_size_limit
        self.zip_size_limit: int = zip_size_limit

    def _check_zip_size(self, zip_file: ZipFile):
        zip_decompressed_size = sum(zi.file_size for zi in zip_file.infolist())
        if zip_decompressed_size > self.zip_size_limit:
            raise ZipTooLargeError(zip_decompressed_size, self.zip_size_limit)

    def _is_file_name_okay(self, name: str):
        return (
            not name.startswith("/")
            and not Path(name).is_absolute()
            and ".." not in name
        )

    def _check_file_names(self, f: ZipFile):
        for n in f.namelist():
            if not self._is_file_name_okay(n):
                raise InvalidFileName(n)

    def _check_decompressed_sizes(self, f: ZipFile):
        for info in f.filelist:
            if info.file_size > self.file_size_limit:
                raise FileTooLargeError(
                    info.filename, info.file_size, self.file_size_limit
                )

    def _check_crc(self, zip_file: ZipFile):
        bad_filename = zip_file.testzip()
        if bad_filename is not None:
            msg = f"{bad_filename} is corrupted (CRC checksum error)!"
            raise InvalidZipError(msg)

    def __call__(self, zip_file: ZipFile):
        self._check_file_names(zip_file)
        self._check_decompressed_sizes(zip_file)
        self._check_zip_size(zip_file)
        self._check_crc(zip_file)
