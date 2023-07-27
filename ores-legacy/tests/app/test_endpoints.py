import os
import pytest
from unittest import mock
import yaml
import json

with open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "app/config/test_available_models.yaml",
    )
) as f:
    available_models = yaml.safe_load(f)

with open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_responses/liftwing_responses.json",
    )
) as f:
    lw_responses = json.load(f)["responses"]

with open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sample_responses/enwiki_ores_responses.json",
    )
) as f:
    ores_responses = json.load(f)


@pytest.mark.asyncio
async def test_read_main(client):
    """
    Test that the root endpoint redirects to /docs
    """
    response = await client.get("/")
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


@mock.patch("app.utils.available_models", available_models)
@pytest.mark.asyncio
async def test_get_scores_no_query_params(client):
    response = await client.get("/v3/scores/dewiki")
    expected = {"dewiki": {"models": {"damaging": {"version": "0.5.1"}}}}
    assert response.status_code == 200
    assert response.json() == expected


@mock.patch("app.utils.available_models", available_models)
@pytest.mark.asyncio
async def test_get_info_for_some_models(client):
    response = await client.get("/v3/scores/cswiki?models=damaging|goodfaith")
    expected = {
        "cswiki": {
            "models": {
                "damaging": {"version": "0.6.0"},
                "goodfaith": {"version": "0.6.0"},
            }
        }
    }
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
    assert response.json()["detail"]["error"]["code"] == "not found"


@pytest.mark.asyncio
async def test_get_scores_returns_404_when_context_does_not_exist(client):
    response = await client.get("/v3/scores/whateverwiki")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_scores_too_many_revisions(client):
    response = await client.get(
        "/v3/scores/enwiki?models=articletopic|articlequality|damaging|goodfaith&revids=949447954|949447964|949447961|949447982|949447986|949448019|949448035|949448037|949448042|949448059|949448053|949448061|949448062|949448069|949448088|949448103|949448113|949448124|949448126|949448134|949448130|949448142|949448155|949448170|949448172|949448188|949448190|949448196|949448224|949448236|949448243|949448247|949448245|949448248|949448259|949448250|949448265|949448274|949448278|949448277|949448283|949448304|949448307|949448316|949448325|949448327|949448343|949448363|949448372|949448378"
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error"]["code"] == "too many requests"


@pytest.mark.asyncio
async def test_get_scores_unsupported_features(client):
    response = await client.get(
        "/v3/scores/enwiki/12345/damaging?model_info=statistics"
    )
    assert response.status_code == 400
