import json
import os
from unittest import mock

import aiohttp
import pytest
from app.liftwing.response import get_liftwing_response, get_lw_namespace

with open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "sample_responses/liftwing_responses.json",
    )
) as f:
    lw_responses = json.load(f)["responses"]


@mock.patch("aiohttp.ClientSession.post")
@pytest.mark.asyncio
async def test_get_liftwing_response(mock_post):
    mock_post.return_value.__aenter__.return_value.status = 200
    mock_post.return_value.__aenter__.return_value.text = json.dumps(
        lw_responses["articlequality"]
    )
    mock_post.return_value.__aenter__.return_value.json.return_value = lw_responses[
        "articlequality"
    ]
    response_dict = await get_liftwing_response(
        db="enwiki",
        model_name="articlequality",
        rev_id=1234,
        features=False,
        liftwing_url="http://dummy.url",
    )
    assert response_dict == lw_responses["articlequality"]


@mock.patch("aiohttp.ClientSession.post")
@pytest.mark.asyncio
async def test_get_liftwing_wp10_decorator_response(mock_post):
    """
    Test that the wp10 decorator works as expected. When the model_name is wp10, the response from
    liftwing is manipulated to match that of articlequality, however model_name should still be wp10.
    """
    mock_post.return_value.__aenter__.return_value.status = 200
    mock_post.return_value.__aenter__.return_value.text = json.dumps(
        lw_responses["articlequality"]
    )
    mock_post.return_value.__aenter__.return_value.json.return_value = lw_responses[
        "articlequality"
    ]
    response_dict = await get_liftwing_response(
        db="enwiki",
        model_name="wp10",
        rev_id=1097728152,  # need the correct rev_id as we search for it in the response
        features=False,
        liftwing_url="http://dummy.url",
    )
    assert response_dict == lw_responses["wp10"]


@mock.patch("aiohttp.ClientSession.post")
@pytest.mark.asyncio
async def test_get_liftwing_response_400_response(mock_post):
    mock_post.return_value.__aenter__.return_value.status = 400
    mock_post.return_value.__aenter__.return_value.reason = "Bad Request"
    mock_post.return_value.__aenter__.return_value.json.return_value = {
        "error": "Some error message"
    }
    response_dict = await get_liftwing_response(
        db="enwiki",
        model_name="articlequality",
        rev_id=11422868611312312,
        features=False,
        liftwing_url="http://dummy.url",
    )
    response_msg = response_dict["enwiki"]["scores"][str(11422868611312312)][
        "articlequality"
    ]
    assert response_msg == {
        "error": {"message": "Some error message", "type": "Bad Request"}
    }


@mock.patch(
    "aiohttp.ClientSession.post",
    side_effect=aiohttp.ClientConnectionError("dummy message"),
)
@pytest.mark.asyncio
async def test_get_liftwing_wrong_url_response(mock_post):
    response_dict = await get_liftwing_response(
        db="enwiki",
        model_name="articlequality",
        rev_id=11422868611312312,
        features=False,
        liftwing_url="http://dummy.url",
    )
    response_msg = response_dict["enwiki"]["scores"][str(11422868611312312)][
        "articlequality"
    ]
    assert response_msg["error"]["type"] == "ClientError"


def test_get_lw_namespace():
    assert get_lw_namespace("itemquality") == "revscoring-articlequality"
    assert get_lw_namespace("itemtopic") == "revscoring-articletopic"
    assert get_lw_namespace("damaging") == "revscoring-editquality-damaging"
    assert get_lw_namespace("goodfaith") == "revscoring-editquality-goodfaith"
    assert get_lw_namespace("reverted") == "revscoring-editquality-reverted"
    assert get_lw_namespace("articlequality") == "revscoring-articlequality"
    assert get_lw_namespace("articletopic") == "revscoring-articletopic"
    assert get_lw_namespace("draftquality") == "revscoring-draftquality"
    assert get_lw_namespace("drafttopic") == "revscoring-drafttopic"
