# real-browser fallback for JS-rendered/interactive sources Firecrawl can't handle
"""MCP Integration spec, Section 2 (Playwright). A real browser, for the
specific subset of sources that need JS rendering, clicking through
filters, or multi-step navigation -- data/fetch_firecrawl_sources.py's
plain scrape can't do any of that, only fetch-and-convert a single
already-rendered page.

Per the spec's own priority order, this is the fallback tool, not the
default -- built last, wired in only for a source that's actually been
confirmed to need it. As of this module's own build: every source this
project currently pulls from (team news pages, ourlads.com's depth
charts) rendered correctly as plain server-side HTML and was already
handled by Firecrawl without needing a real browser -- confirmed by
testing both against Firecrawl directly before reaching for this module.
Nothing in this pipeline currently calls scrape_rendered_page() for that
reason; it's here, tested, and ready for whichever future source
actually turns out to need it, not wired into weekly_pipeline.sh on
spec, since that would mean running a full browser for sources that
don't need one -- exactly what the spec says not to do.

Same robots.txt/ToS ground rule as the Firecrawl module: only point this
at sources that are confirmed to allow it, never at login-gated or
paywalled pages (sportsbook account-specific odds pages, PFF grades,
etc.).
"""

from playwright.sync_api import sync_playwright

DEFAULT_TIMEOUT_MS = 20000


def scrape_rendered_page(url: str, wait_for_selector: str | None = None,
                          click_selectors: list[str] | None = None) -> str:
    """Loads `url` in a real (headless) browser, optionally clicking
    through `click_selectors` in order (e.g. a team filter, a week
    selector) before extracting the page's own rendered text -- for a
    page that returns nothing useful to a plain HTTP fetch because its
    real content only exists after JavaScript runs. `wait_for_selector`
    blocks until that element appears, so this doesn't grab a still-
    loading skeleton page. Returns the page's visible text content, the
    same shape data/fetch_firecrawl_sources.py's markdown output is
    meant to feed into -- a caller parses this the same way it would
    parse a Firecrawl result."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, timeout=DEFAULT_TIMEOUT_MS, wait_until="domcontentloaded")
            for selector in click_selectors or []:
                page.click(selector, timeout=DEFAULT_TIMEOUT_MS)
            if wait_for_selector:
                page.wait_for_selector(wait_for_selector, timeout=DEFAULT_TIMEOUT_MS)
            return page.inner_text("body")
        finally:
            browser.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ad-hoc test: render a page and print its text.")
    parser.add_argument("url")
    parser.add_argument("--wait-for", default=None)
    args = parser.parse_args()
    print(scrape_rendered_page(args.url, wait_for_selector=args.wait_for))


if __name__ == "__main__":
    main()
