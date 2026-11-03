from __future__ import annotations

import pandas as pd

from csc869.report import _read


def test_read_missing_file_returns_empty(tmp_path):
    assert _read(tmp_path, "does_not_exist.csv").empty


def test_read_header_only_file_returns_empty(tmp_path):
    # A stage can legitimately write a header-only CSV (for example when the
    # Adult test split has no unseen categories); the report must not crash.
    (tmp_path / "unseen_categories.csv").write_text("\n", encoding="utf-8")
    assert _read(tmp_path, "unseen_categories.csv").empty


def test_read_normal_file(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_csv(tmp_path / "table.csv", index=False)
    table = _read(tmp_path, "table.csv")
    assert list(table["a"]) == [1, 2]
