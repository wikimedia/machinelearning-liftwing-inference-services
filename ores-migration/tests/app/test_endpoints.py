import pytest
from unittest import mock
import yaml
import json
from app.main import app
from httpx import AsyncClient


@pytest.fixture
def client():
    return AsyncClient(app=app, base_url="http://test")


with open("tests/app/config/test_available_models.yaml") as f:
    available_models = yaml.safe_load(f)

with open("tests/sample_responses/liftwing_responses.json") as f:
    lw_responses = json.load(f)["responses"]

with open("tests/sample_responses/enwiki_ores_responses.json") as f:
    ores_responses = json.load(f)


@pytest.mark.asyncio
async def test_read_main(client):
    response = await client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ORES/LiftWing calls legacy service"}


@mock.patch("app.utils.available_models", available_models)
@pytest.mark.asyncio
async def test_get_scores_no_query_params(client):
    response = await client.get("/v3/scores/dewiki")
    expected = {"dewiki": {"models": {"damaging": {"version": "0.5.1"}}}}
    assert response.status_code == 200
    assert response.json() == expected


@mock.patch("app.liftwing.response.get_liftwing_response")
@pytest.mark.asyncio
async def test_get_scores_returns_scores_for_one_revid(mock_response, client):
    mock_response.return_value.json = lw_responses["articlequality"]
    response = await client.get(
        "/v3/scores/enwiki?models=articlequality&revids=1097728152"
    )
    assert "articlequality" in response.json()["enwiki"]["models"]


@mock.patch("app.liftwing.response.get_liftwing_response")
@pytest.mark.asyncio
async def test_get_scores_returns_scores_multiple_revids(mock_get_response, client):
    mock_get_response.side_effect = [
        lw_responses["articlequality"],
        lw_responses["articlequality_2"],
    ]
    response = await client.get(
        "/v3/scores/enwiki?models=articlequality&revids=1097728152|1142286861"
    )
    ores_response = ores_responses["context_many_rev_ids"]["response"]
    assert response.status_code == 200
    assert response.json() == ores_response
    assert "articlequality" in response.json()["enwiki"]["models"]


@mock.patch("app.utils.available_models", available_models)
@mock.patch("app.liftwing.response.get_liftwing_response")
@pytest.mark.asyncio
async def test_get_scores_returns_scores_revid_without_model(mock_get_response, client):
    mock_get_response.side_effect = [
        lw_responses["articlequality"],
        lw_responses["draftquality"],
    ]
    response = await client.get("/v3/scores/enwiki?revids=1097728152")
    ores_response = ores_responses["context_one_rev_ids_no_models"]["response"]
    assert response.status_code == 200
    assert response.json() == ores_response
    assert "articlequality" in response.json()["enwiki"]["models"]


@pytest.mark.asyncio
async def test_get_scores_returns_400_when_models_do_not_exist_for_context(client):
    response = await client.get("/v3/scores/enwiki?models=NONEXISTENTMODEL")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_scores_returns_404_when_context_does_not_exist(client):
    response = await client.get("/v3/scores/whateverwiki")
    assert response.status_code == 404
