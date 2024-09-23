from __future__ import annotations

from dataclasses import dataclass
from functools import total_ordering


@dataclass
@total_ordering
class Version:
    version_str: str

    def _split(self):
        return list(self.version_str.split("."))

    @property
    def major(self) -> int:
        return int(self._split()[0])

    @property
    def minor(self) -> int:
        return int(self._split()[1])

    @property
    def patch(self) -> int | str:
        patchval = self._split()[2]
        try:
            return int(patchval)
        except ValueError:
            return patchval

    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"Version({self.version_str})"

    def __eq__(self, value: Version) -> bool:
        return self._split() == value._split()

    def __lt__(self, value: Version) -> bool:
        if not isinstance(value, Version):
            return NotImplemented

        for self_part, value_part in zip([self.major, self.minor], [value.major, value.minor]):
            if self_part > value_part:
                return False
            elif self_part < value_part:
                return True

        if not isinstance(self.patch, str) and not isinstance(value.patch, str):
            return self.patch < value.patch
        return False

    def __gt__(self, value: Version) -> bool:
        if not isinstance(value, Version):
            return NotImplemented

        for self_part, value_part in zip([self.major, self.minor], [value.major, value.minor]):
            if self_part > value_part:
                return True
            elif self_part < value_part:
                return False

        if not isinstance(self.patch, str) and not isinstance(value.patch, str):
            return self.patch > value.patch
        return False
