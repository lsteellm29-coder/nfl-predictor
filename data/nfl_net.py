# retries for nfl_data_py's downloads
"""nfl_data_py downloads everything from GitHub releases / a CSV host with no retry, and its
own error handling is broken (`except Error` is an undefined name, so ANY failure while reading
a play-by-play year surfaces as an unrelated NameError). One dropped connection, DNS blip or 504
therefore killed a whole weekly run -- and this machine had all three in a single morning.

install_retries() wraps every `import_*` function once, retrying transient failures with a
growing delay. Permanent failures are NOT retried: a 404 (e.g. the pbp_participation file that
doesn't exist yet for a new season -- data/pbp_loader.py relies on failing fast there) or any
error that isn't a network failure raises immediately. config.py installs it at import, so every
entry point gets it.
"""

import functools
import http.client
import socket
import time
import urllib.error

import nfl_data_py as nfl

ATTEMPTS = 4
DELAY_SECONDS = 5
TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}


def _chain(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def is_transient(error: BaseException) -> bool:
    """True for a network failure worth retrying. Looks through the exception chain, because
    nfl_data_py re-raises whatever went wrong as a NameError from inside its `except` block."""
    for e in _chain(error):
        if isinstance(e, urllib.error.HTTPError):        # before URLError: it is a subclass
            return e.code in TRANSIENT_HTTP_CODES
        if isinstance(e, (urllib.error.URLError, ConnectionError, TimeoutError, socket.gaierror,
                          http.client.HTTPException)):
            return True
    return False


def retrying(fn, attempts: int = ATTEMPTS, delay: float = DELAY_SECONDS, sleep=time.sleep):
    if getattr(fn, "_retrying", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(1, attempts + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == attempts or not is_transient(e):
                    raise
                wait = delay * attempt
                print(f"  {fn.__name__}: network error ({type(e).__name__}); retry {attempt}/{attempts - 1} in {wait:.0f}s")
                sleep(wait)

    wrapper._retrying = True
    return wrapper


def install_retries(module=nfl) -> list[str]:
    """Wrap each import_* function on `module` (default nfl_data_py). Idempotent."""
    wrapped = []
    for name in dir(module):
        if name.startswith("import_") and callable(getattr(module, name)):
            fn = getattr(module, name)
            if not getattr(fn, "_retrying", False):
                setattr(module, name, retrying(fn))
                wrapped.append(name)
    return wrapped
