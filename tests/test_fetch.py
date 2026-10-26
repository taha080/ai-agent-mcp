"""HTTP fetch with a mocked transport. No network and no DNS."""

from __future__ import annotations

import httpx
import pytest

from ai_agent_mcp.tools.fetch import FetchError, fetch_page, html_to_text, validate_public_http_url

_HTML = """
<html>
  <head>
    <style>body { color: red; }</style>
    <script>alert("secret")</script>
  </head>
  <body>
    <h1>Title</h1>
    <p>Hello <b>world</b>.</p>
    <ul><li>One</li><li>Two</li></ul>
  </body>
</html>
"""


def _client_page(
    body: str, *, status: int = 200, content_type: str = "text/html"
) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    return httpx.MockTransport(handler)


def test_html_is_cleaned_and_scripts_removed() -> None:
    text = fetch_page("https://example.com/docs", transport=_client_page(_HTML))
    assert "Title" in text
    assert "Hello world." in text
    assert "- One" in text
    assert "alert" not in text
    assert "color" not in text


def test_html_helper_collapses_whitespace() -> None:
    assert "Hello world." in html_to_text("<p>Hello   <b>world</b>.</p>")


def test_plain_text_is_returned() -> None:
    transport = _client_page("alpha\nbeta", content_type="text/plain")
    assert fetch_page("https://example.com/note.txt", transport=transport) == "alpha\nbeta"


def test_json_is_returned_as_text() -> None:
    transport = _client_page('{"ok": true}', content_type="application/json")
    assert '"ok": true' in fetch_page("https://example.com/data.json", transport=transport)


def test_truncates_large_bodies() -> None:
    transport = _client_page("abcdefghijKLMNOP", content_type="text/plain")
    text = fetch_page("https://example.com/big", transport=transport, max_bytes=10)
    assert text.startswith("abcdefghij")
    assert "truncated to 10 bytes" in text
    assert "KLMNOP" not in text


def test_http_error_status() -> None:
    transport = _client_page("missing", status=404, content_type="text/plain")
    with pytest.raises(FetchError, match="404"):
        fetch_page("https://example.com/missing", transport=transport)


def test_timeout_becomes_fetch_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    with pytest.raises(FetchError, match="Timed out"):
        fetch_page("https://example.com", transport=httpx.MockTransport(handler))


def test_refuses_non_text_content() -> None:
    transport = _client_page("not really", content_type="image/png")
    with pytest.raises(FetchError, match="non-text"):
        fetch_page("https://example.com/pixel.png", transport=transport)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/plain,hello",
        "http://127.0.0.1/admin",
        "http://10.1.2.3/",
        "http://192.168.0.5/",
        "http://169.254.169.254/latest/meta-data",
        "http://localhost/secret",
        "http://[::1]/",
        "http://2130706433/",
        "ftp://example.com/file",
        "",
    ],
)
def test_refuses_unsafe_urls(url: str) -> None:
    with pytest.raises(FetchError):
        validate_public_http_url(url)


def test_blocks_redirect_to_loopback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        return httpx.Response(200, text="secret", headers={"content-type": "text/plain"})

    with pytest.raises(FetchError, match="non-public"):
        fetch_page("https://example.com/start", transport=httpx.MockTransport(handler))


def test_blocks_hostname_that_resolves_to_a_private_address() -> None:
    called = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, text="nope")

    with pytest.raises(FetchError, match="non-public"):
        fetch_page(
            "https://example.com/",
            transport=httpx.MockTransport(handler),
            resolver=lambda _host: ["10.0.0.5"],
        )
    assert called is False


def test_allows_a_public_resolution_with_mock_transport() -> None:
    transport = _client_page("hello", content_type="text/plain")
    text = fetch_page(
        "https://example.com/hello",
        transport=transport,
        resolver=lambda _host: ["93.184.216.34"],
    )
    assert text == "hello"
