def page_slice(items, page, per_page):
    """Return the items on a 1-indexed page.

    Raises ValueError for non-positive page or per_page. Pages past the end are empty.
    """
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be positive")
    start = (page - 1) * per_page
    return items[start : start + per_page]
