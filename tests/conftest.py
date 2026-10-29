import pytest

from agent_eval.engine import ensure_image, get_client


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: needs a reachable container runtime")


@pytest.fixture(scope="session")
def docker_client():
    try:
        client = get_client()
    except RuntimeError as exc:
        pytest.skip(f"no container runtime: {exc}")
    ensure_image(client)
    return client
