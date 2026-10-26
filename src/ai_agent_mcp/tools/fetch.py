"""Fetch a public HTTP(S) URL and return readable text.

Limits are intentional: a timeout, a response-size cap, a redirect cap, and
a refusal to call non-HTTP schemes or non-public addresses. HTML is reduced
to text with the standard library. JavaScript is not executed.

DNS lookups run only for a live fetch. Tests pass an ``httpx`` transport and
therefore stay offline. Pass ``resolver`` to exercise the address check
without contacting a real DNS server.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from html.parser import HTMLParser
from urllib.parse import ParseResult, urlparse

import httpx

from ai_agent_mcp import __version__
from ai_agent_mcp.tools.errors import ToolFailure

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_MAX_BYTES = 500_000
MAX_REDIRECTS = 5
MAX_URL_LENGTH = 2_048
USER_AGENT = f"ai-agent-mcp/{__version__} (+https://github.com/taha080/ai-agent-mcp)"

_BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}
_BLOCKED_SUFFIXES = (".localhost", ".local", ".internal")
_TEXT_TYPES = {
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/xhtml+xml",
    "application/xml",
    "text/css",
    "text/html",
    "text/javascript",
    "text/plain",
    "text/xml",
}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}
_BREAK_TAGS = {
    "article",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "p",
    "section",
    "table",
    "tr",
}

Resolver = Callable[[str], list[str]]


class FetchError(ToolFailure):
    """The URL was refused or the page could not be read."""


class _HTMLText(HTMLParser):
    """Collect visible text and drop script, style, and svg subtrees."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in _BREAK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
            return
        if self._skip:
            return
        if tag in _BREAK_TAGS or tag == "li":
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Return visible text from an HTML document."""
    parser = _HTMLText()
    parser.feed(html)
    parser.close()
    return _collapse_lines("".join(parser.parts))


def fetch_page(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    transport: httpx.BaseTransport | None = None,
    resolver: Resolver | None = None,
) -> str:
    """Fetch ``url`` and return cleaned text.

    Args:
        url: An ``http`` or ``https`` URL.
        timeout: Whole-request timeout in seconds.
        max_bytes: Maximum response body retained.
        transport: ``httpx`` transport. Tests use ``MockTransport``.
        resolver: Maps a hostname to IP strings. Live fetches use DNS when
            this is omitted and ``transport`` is also omitted.

    Raises:
        FetchError: The URL is refused or the request fails.
    """
    validate_public_http_url(url)
    if timeout <= 0:
        raise FetchError("Timeout must be positive.")
    if max_bytes < 1:
        raise FetchError("max_bytes must be at least 1.")

    def guard(request: httpx.Request) -> None:
        parsed = validate_public_http_url(str(request.url))
        host = parsed.hostname or ""
        active = resolver
        if active is None and transport is None:
            active = _dns_resolve
        if active is not None:
            _enforce_resolved(host, active)

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json,text/plain;q=0.9,*/*;q=0.1",
    }
    try:
        with (
            httpx.Client(
                transport=transport,
                timeout=httpx.Timeout(timeout),
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                event_hooks={"request": [guard]},
                headers=headers,
            ) as client,
            client.stream("GET", url) as response,
        ):
            _raise_for_status(response, url)
            media = _media_type(response.headers.get("content-type", ""))
            if not _is_textual(media):
                label = media or "unknown"
                raise FetchError(f"Refusing non-text content type {label}.")
            raw, truncated = _read_limited(response, max_bytes)
            content_type = response.headers.get("content-type", "")
    except FetchError:
        raise
    except httpx.TimeoutException as exc:
        raise FetchError(f"Timed out fetching {url}.") from exc
    except httpx.HTTPError as exc:
        cause = exc.__cause__
        if isinstance(cause, FetchError):
            raise cause from exc
        raise FetchError(f"Could not fetch {url}: {exc.__class__.__name__}.") from exc

    if not raw:
        return "The response was empty."
    if b"\x00" in raw[:1024]:
        raise FetchError("Response is not text.")
    text = _decode(raw, content_type)
    text = html_to_text(text) if _is_html(media, text) else _clean_plain(text)
    if not text:
        return "The response had no readable text."
    if truncated:
        text = f"{text}\n\n[truncated to {max_bytes} bytes]"
    return text


def validate_public_http_url(url: str) -> ParseResult:
    """Reject schemes and hosts that must never be fetched. Does not use DNS.

    Returns the ``urllib.parse`` result so callers can read the hostname.
    """
    if not isinstance(url, str):
        raise FetchError("URL must be a string.")
    candidate = url.strip()
    if not candidate:
        raise FetchError("URL is empty.")
    if len(candidate) > MAX_URL_LENGTH:
        raise FetchError("URL is too long.")
    parsed = urlparse(candidate)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise FetchError("Only http and https URLs can be fetched.")
    host = parsed.hostname
    if not host:
        raise FetchError("URL is missing a host.")
    lowered = host.lower().rstrip(".")
    if lowered in _BLOCKED_HOSTS or lowered.endswith(_BLOCKED_SUFFIXES):
        raise FetchError(f"Refusing host {host!r}.")
    literal = _literal_ip(lowered)
    if literal is not None:
        _reject_ip(literal)
    return parsed


def _enforce_resolved(host: str, resolver: Resolver) -> None:
    if _literal_ip(host) is not None:
        return
    try:
        addresses = resolver(host)
    except FetchError:
        raise
    except Exception as exc:
        raise FetchError(f"Could not resolve host {host!r}.") from exc
    if not addresses:
        raise FetchError(f"Could not resolve host {host!r}.")
    for address in addresses:
        _reject_address(address)


def _dns_resolve(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchError(f"Could not resolve host {host!r}.") from exc
    return [str(info[4][0]) for info in infos]


def _literal_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return _unwrap(ipaddress.ip_address(host))
    except ValueError:
        pass
    if host.isdigit():
        try:
            return ipaddress.ip_address(int(host))
        except ValueError:
            return None
    return None


def _reject_address(address: str) -> None:
    cleaned = address.split("%", 1)[0]
    try:
        ip = _unwrap(ipaddress.ip_address(cleaned))
    except ValueError as exc:
        raise FetchError(f"Resolved address {address!r} is not valid.") from exc
    _reject_ip(ip)


def _unwrap(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    mapped = getattr(ip, "ipv4_mapped", None)
    if isinstance(mapped, ipaddress.IPv4Address):
        return mapped
    return ip


def _reject_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise FetchError(f"Refusing non-public address {ip}.")


def _raise_for_status(response: httpx.Response, url: str) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise FetchError(f"HTTP {exc.response.status_code} for {url}.") from exc


def _read_limited(response: httpx.Response, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        remaining = max_bytes - total
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            return b"".join(chunks), True
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), False


def _decode(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    for part in content_type.split(";")[1:]:
        key, _, value = part.strip().partition("=")
        if key.lower() == "charset" and value:
            charset = value.strip().strip('"')
            break
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _is_textual(media: str) -> bool:
    if not media:
        return True
    if media in _TEXT_TYPES or media.startswith("text/"):
        return True
    return media.endswith("+json") or media.endswith("+xml")


def _is_html(media: str, text: str) -> bool:
    if media in {"text/html", "application/xhtml+xml"}:
        return True
    if media:
        return False
    sample = text.lstrip()[:200].lower()
    return sample.startswith(("<!doctype html", "<html", "<head", "<body"))


def _clean_plain(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _collapse_lines(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()
