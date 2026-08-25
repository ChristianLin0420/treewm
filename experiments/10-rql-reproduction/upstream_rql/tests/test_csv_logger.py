import csv

import pytest

from utils.csv_logger import CsvLogger


def read_rows(path):
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def test_csv_logger_appends_without_duplicate_header_or_step(tmp_path):
    path = tmp_path / 'train.csv'
    logger = CsvLogger(path)
    assert logger.log({'metric': 'value,with,commas'}, step=5)
    logger.close()

    resumed = CsvLogger(path)
    assert not resumed.log({'metric': 'must-not-overwrite'}, step=5)
    assert resumed.log({'metric': 'second'}, step=10)
    resumed.close()

    text = path.read_text()
    assert text.count('metric,step') == 1
    assert read_rows(path) == [
        {'metric': 'value,with,commas', 'step': '5'},
        {'metric': 'second', 'step': '10'},
    ]


def test_csv_logger_preserves_existing_bytes_and_repairs_missing_newline(tmp_path):
    path = tmp_path / 'eval.csv'
    original = b'metric,step\nold,1'
    path.write_bytes(original)

    logger = CsvLogger(path)
    assert logger.log({'metric': 'new'}, step=2)
    logger.close()

    assert path.read_bytes().startswith(original + b'\n')
    assert [row['step'] for row in read_rows(path)] == ['1', '2']


def test_csv_logger_fails_loudly_on_schema_drift(tmp_path):
    path = tmp_path / 'eval.csv'
    with CsvLogger(path) as logger:
        logger.log({'metric': 1.0}, step=1)
    resumed = CsvLogger(path)
    with pytest.raises(ValueError, match='schema changed'):
        resumed.log({'metric': 2.0, 'unexpected': 3.0}, step=2)
    resumed.close()
    assert [row['step'] for row in read_rows(path)] == ['1']

