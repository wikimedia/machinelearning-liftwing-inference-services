# import json
# import pytest
# from unittest import mock
# from app.liftwing.response import get_liftwing_response
#
# with open("tests/sample_responses/liftwing_responses.json") as f:
#     responses = json.load(f)["responses"]
#
#
# @mock.patch("requests.post")
# async def test_get_liftwing_response(mock_post):
#     mock_post.return_value.status_code = 200
#     mock_post.return_value.text = json.dumps(responses["articlequality"])
#     mock_post.return_value.json.return_value = responses["articlequality"]
#     response_dict = get_liftwing_response(
#         env="production",
#         db="enwiki",
#         model_name="articlequality",
#         rev_id=1234,
#         model_hostname="revscoring-articlequality",
#     )
#     print(response_dict)
#     pass
