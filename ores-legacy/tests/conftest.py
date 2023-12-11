import pytest
from app.main import app
from httpx import AsyncClient


@pytest.fixture(scope="session")
def client():
    return AsyncClient(app=app, base_url="http://test")
