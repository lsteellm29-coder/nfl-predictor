# shared Odds API error handling
"""requests' own raise_for_status() message includes the full request URL --
and every Odds API request carries `?apiKey=<the key>` -- so an unhandled 401 (a
deactivated key, say) printed the key into terminals, the automation logs, and
any pasted traceback. This raises the same exception family (requests.HTTPError,
so existing `except requests.RequestException` handlers still work) with the
status and The Odds API's own error code, and nothing else.
"""

import requests


class OddsAPIError(requests.HTTPError):
    pass


def odds_get(url: str, params: dict, timeout: float = 15) -> requests.Response:
    """requests.get for an Odds API endpoint, with every failure raised as an OddsAPIError that
    can't contain the key. requests puts the full URL -- ?apiKey=... included -- into a
    connection/DNS/timeout error's message too, not just into an HTTP-status error's, so those are
    caught here as well, and raised `from None` so the original never appears in a traceback."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise OddsAPIError(f"The Odds API request failed ({type(e).__name__})") from None
    raise_for_odds_status(resp)
    return resp


def raise_for_odds_status(resp: requests.Response) -> None:
    if resp.status_code < 400:
        return
    try:
        code = resp.json().get("error_code")
    except ValueError:
        code = None
    detail = f" ({code})" if code else ""
    raise OddsAPIError(f"The Odds API returned HTTP {resp.status_code}{detail}", response=resp)
