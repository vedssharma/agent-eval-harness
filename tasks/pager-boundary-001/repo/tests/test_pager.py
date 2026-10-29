import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pager import page_slice


def test_first_page():
    assert page_slice([1, 2, 3, 4, 5], 1, 2) == [1, 2]


def test_last_partial_page():
    assert page_slice([1, 2, 3, 4, 5], 3, 2) == [5]
