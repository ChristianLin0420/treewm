"""Dependency-free, append-only CSV logging for resumable jobs."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Mapping


class CsvLogger:
    """Append rows once per absolute step without rewriting prior output."""

    def __init__(self, path: os.PathLike[str] | str):
        self.path = str(path)
        self.header: list[str] | None = None
        self.file = None
        self.writer = None
        self.existing_steps: set[str] = set()
        self._needs_leading_newline = False
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, newline='') as existing_file:
                reader = csv.DictReader(existing_file)
                self.header = reader.fieldnames
                if not self.header or 'step' not in self.header:
                    raise ValueError(f'Existing CSV has no valid step header: {self.path}')
                for row in reader:
                    if row.get('step'):
                        self.existing_steps.add(row['step'])
            with open(self.path, 'rb') as existing_file:
                existing_file.seek(-1, os.SEEK_END)
                self._needs_leading_newline = existing_file.read(1) not in (b'\n', b'\r')

    def log(self, row: Mapping[str, Any], step: int) -> bool:
        """Append ``row`` and return false when that absolute step already exists."""

        step_key = str(step)
        if step_key in self.existing_steps:
            return False
        output_row = dict(row)
        output_row['step'] = step
        if self.file is None:
            Path(self.path).resolve().parent.mkdir(parents=True, exist_ok=True)
            self.file = open(self.path, 'a', newline='')
            if self._needs_leading_newline:
                self.file.write('\n')
            if self.header is None:
                self.header = list(output_row)
            self.writer = csv.DictWriter(self.file, fieldnames=self.header, extrasaction='raise')
            if os.path.getsize(self.path) == 0:
                self.writer.writeheader()
        unknown_keys = set(output_row) - set(self.header)
        if unknown_keys:
            raise ValueError(f'CSV schema changed for {self.path}: {sorted(unknown_keys)}')
        self.writer.writerow(output_row)
        self.file.flush()
        self.existing_steps.add(step_key)
        return True

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

