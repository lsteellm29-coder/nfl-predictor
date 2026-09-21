# The suite must never touch the network. A test that quietly did (the HTML-escaping
# test looked up team names through nfl_data_py) passed on a good connection and then
# failed the whole weekly pipeline the day the machine was offline -- and, worse,
# would exercise live data instead of the fixture it thought it was using.
# Any real socket use now fails the test loudly, wherever it runs.
import socket

import pytest


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError("tests must not use the network -- monkeypatch the fetch instead")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
