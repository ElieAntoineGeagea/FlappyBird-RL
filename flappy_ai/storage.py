"""Durable output helpers; run folders never replace earlier experiments."""
import csv
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def unique_directory(parent, prefix):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = parent / f'{prefix}_{stamp}_{uuid4().hex[:8]}'
    path.mkdir(exist_ok=False)
    return path


def replace_with_retry(source, destination):
    """Bounded retry for transient Windows file-sharing locks."""
    for attempt in range(7):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.01 * 2 ** attempt)


def atomic_json(path, data):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CSVJournal:
    """Exclusive creation and a durable flush after each completed episode."""
    def __init__(self, path, fields):
        self.file = Path(path).open('x', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.flush()

    def flush(self):
        self.file.flush()
        os.fsync(self.file.fileno())

    def append(self, row):
        self.writer.writerow(row)
        self.flush()

    def close(self):
        self.file.close()


def save_comparison(row, parent='artifacts/comparisons'):
    directory = Path(parent)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    path = directory / f'comparison_{stamp}_{uuid4().hex[:8]}.csv'
    journal = CSVJournal(path, list(row))
    try:
        journal.append(row)
    finally:
        journal.close()
    return path
