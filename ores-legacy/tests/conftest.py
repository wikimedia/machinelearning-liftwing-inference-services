import pytest
from httpx import AsyncClient
from app.main import app


@pytest.fixture(scope="session")
def client():
    return AsyncClient(app=app, base_url="http://test")
