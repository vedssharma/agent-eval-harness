import sys
from pathlib import Path

import pytest

# Tests run with the repo root as cwd (/workspace in the sandbox).
sys.path.insert(0, str(Path.cwd() / "src"))

from pager import page_slice


@pytest.mark.parametrize("page", [0, -1])
def test_non_positive_page_raises(page):
    with pytest.raises(ValueError):
        page_slice([1, 2, 3], page, 2)


@pytest.mark.parametrize("per_page", [0, -3])
def test_non_positive_per_page_raises(per_page):
    with pytest.raises(ValueError):
        page_slice([1, 2, 3], 1, per_page)


def test_past_end_is_empty():
    assert page_slice([1, 2, 3], 5, 2) == []


def test_exact_boundary():
    assert page_slice([1, 2, 3, 4], 2, 2) == [3, 4]
    assert page_slice([1, 2, 3, 4], 3, 2) == []
