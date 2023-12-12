import pytest

from python.api_utils import get_rest_endpoint_page_summary


def test_get_rest_endpoint_liftwing():
    """
    Test the get_rest_endpoint_page_summary function with a rest_gateway_endpoint as it would
    happen on Lift Wing.
    """
    rest_gateway_endpoint = "http://rest-gateway.discovery.wmnet:4113"
    mw_host = "https://en.wikipedia.org"
    mw_host_header = "en.wikipedia.org"
    title = "Dummypage"

    endpoint = get_rest_endpoint_page_summary(
        rest_gateway_endpoint, mw_host, mw_host_header, title
    )
    expected_url = (
        "http://rest-gateway.discovery.wmnet:4113/en.wikipedia.org/v1/page/summary"
        "/Dummypage"
    )
    assert endpoint == expected_url


def test_get_rest_endpoint_local_run():
    """
    Test the get_rest_endpoint_page_summary function without a rest_gateway_endpoint as it would
    happen during a run when deployed outside WMF production cluster
    The same behavior is expected when running locally.
    """
    rest_gateway_endpoint = None
    mw_host = "https://en.wikipedia.org"
    mw_host_header = "en.wikipedia.org"
    title = "Clandonald"
    endpoint = get_rest_endpoint_page_summary(
        rest_gateway_endpoint, mw_host, mw_host_header, title
    )
    expected_url = "https://en.wikipedia.org/api/rest_v1/page/summary/Clandonald"
    assert endpoint == expected_url


if __name__ == "__main__":
    pytest.main()
