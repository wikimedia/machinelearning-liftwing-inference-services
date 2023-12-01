from furl import furl


def get_rest_endpoint_page_summary(
    rest_gateway_endpoint: str, mw_host: str, mw_host_header: str, title: str
):
    """
    Get the REST endpoint for the page summary API.
    """
    if rest_gateway_endpoint:
        base_url = furl(rest_gateway_endpoint).join(mw_host_header)
        path_prefix = "v1"
    else:
        base_url = furl(mw_host)
        path_prefix = "api/rest_v1"
    base_url.path = base_url.path / path_prefix / "page" / "summary" / title
    return base_url.url
