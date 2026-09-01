"""Tests for the QuEL-3 protobuf HTTP transport."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from http.client import HTTPMessage, IncompleteRead
from threading import Event
from types import TracebackType
from typing import Any, cast
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request

import pytest
from grpclib.const import Cardinality, Status
from grpclib.exceptions import GRPCError, StreamTerminatedError

from qubex.backend.quel3.infra import quelware_http_transport as transport_module
from qubex.backend.quel3.infra.quelware_http_transport import ProtobufHttpChannel
from qubex.backend.quel3.infra.quelware_transport_config import HttpScheme


class _RequestMessage:
    def __bytes__(self) -> bytes:
        return b"request-body"


class _ResponseMessage:
    @staticmethod
    def parse(body: bytes) -> bytes:
        return body


class _HttpResponse:
    def __init__(
        self,
        body: bytes,
        content_type: str,
        *,
        read_error: Exception | None = None,
    ) -> None:
        self._body = body
        self._read_error = read_error
        self.headers = HTTPMessage()
        self.headers["content-type"] = content_type

    def __enter__(self) -> _HttpResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def close(self) -> None:
        return None


def _patch_channel_opener(
    monkeypatch: pytest.MonkeyPatch,
    open_request: Callable[..., _HttpResponse],
) -> None:
    class _Opener:
        def open(
            self,
            request: Request,
            *,
            timeout: float | None,
        ) -> _HttpResponse:
            return open_request(request, timeout=timeout)

    monkeypatch.setattr(
        transport_module,
        "build_opener",
        lambda *handlers: _Opener(),
    )


def _request(channel: ProtobufHttpChannel, metadata: dict[str, str]) -> bytes:
    async def _invoke() -> bytes:
        stream = channel.request(
            "/quelware.Service/Call",
            Cardinality.UNARY_UNARY,
            _RequestMessage,
            _ResponseMessage,
            metadata=metadata,
        )
        async with stream:
            await stream.send_message(_RequestMessage(), end=True)
            return await stream.recv_message()

    return asyncio.run(_invoke())


def test_https_channel_adds_secret_headers_after_rpc_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secret headers should override conflicting RPC metadata."""
    captured: dict[str, object] = {}

    def _urlopen(
        request: Request,
        *,
        timeout: float | None,
    ) -> _HttpResponse:
        captured.update(request=request, timeout=timeout)
        return _HttpResponse(b"response-body", "application/protobuf")

    _patch_channel_opener(monkeypatch, _urlopen)
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme="https",
        base_path="api",
        default_timeout_seconds=12.5,
        secret_headers={
            "cf-access-client-id": "configured-id",
            "cf-access-client-secret": "configured-secret",
        },
    )

    response = _request(
        channel,
        {
            "cf-access-client-id": "metadata-id",
            "cf-access-client-secret": "metadata-secret",
        },
    )

    request = captured["request"]
    assert isinstance(request, Request)
    headers = {name.lower(): value for name, value in request.header_items()}
    assert request.full_url == "https://api.example.com:443/api/quelware.Service/Call"
    assert request.data == b"request-body"
    assert headers["cf-access-client-id"] == "configured-id"
    assert headers["cf-access-client-secret"] == "configured-secret"
    assert captured["timeout"] == 12.5
    assert response == b"response-body"


def test_https_channel_omits_unspecified_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unspecified HTTPS port should use the scheme default without an empty colon."""
    captured: dict[str, object] = {}

    def _urlopen(
        request: Request,
        *,
        timeout: float | None,
    ) -> _HttpResponse:
        captured["request"] = request
        return _HttpResponse(b"response-body", "application/protobuf")

    _patch_channel_opener(monkeypatch, _urlopen)
    channel = ProtobufHttpChannel("api.example.com", None, scheme="https")

    _request(channel, {})

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "https://api.example.com/quelware.Service/Call"


def test_http_channel_rejects_secret_headers() -> None:
    """Secret headers should never be sent over unencrypted HTTP."""
    with pytest.raises(ValueError, match="HTTPS"):
        ProtobufHttpChannel(
            "localhost",
            8080,
            scheme="http",
            secret_headers={"x-api-key": "configured-secret"},
        )


def test_http_channel_posts_to_unencrypted_debug_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTP transport should post protobuf requests to the configured debug endpoint."""
    captured: dict[str, object] = {}

    def _urlopen(
        request: Request,
        *,
        timeout: float | None,
    ) -> _HttpResponse:
        captured.update(request=request, timeout=timeout)
        return _HttpResponse(b"response-body", "application/protobuf")

    _patch_channel_opener(monkeypatch, _urlopen)
    channel = ProtobufHttpChannel(
        "localhost",
        8080,
        scheme="http",
        base_path="debug",
    )

    response = _request(channel, {})

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "http://localhost:8080/debug/quelware.Service/Call"
    assert response == b"response-body"


def test_http_channel_treats_slash_base_path_as_origin_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slash base path should not add an empty path segment before the RPC route."""
    captured: dict[str, object] = {}

    def _urlopen(
        request: Request,
        *,
        timeout: float | None,
    ) -> _HttpResponse:
        captured["request"] = request
        return _HttpResponse(b"response-body", "application/protobuf")

    _patch_channel_opener(monkeypatch, _urlopen)
    channel = ProtobufHttpChannel(
        "localhost",
        8080,
        scheme="http",
        base_path="/",
    )

    _request(channel, {})

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "http://localhost:8080/quelware.Service/Call"


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_channel_uses_explicit_authenticated_proxy(
    monkeypatch: pytest.MonkeyPatch,
    scheme: HttpScheme,
) -> None:
    """An explicit proxy URL should use a channel-local opener with Basic credentials."""
    captured: dict[str, object] = {}

    class _ProxyOpener:
        def open(
            self,
            request: Request,
            *,
            timeout: float | None,
        ) -> _HttpResponse:
            captured.update(request=request, timeout=timeout)
            return _HttpResponse(b"response-body", "application/protobuf")

    def _build_opener(*handlers: object) -> _ProxyOpener:
        captured["handlers"] = handlers
        return _ProxyOpener()

    monkeypatch.setattr(transport_module, "build_opener", _build_opener)
    proxy_url = "http://proxy-user:proxy-password@proxy.example.com:3128"
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme=scheme,
        default_timeout_seconds=7.5,
        proxy_url=proxy_url,
    )

    response = _request(channel, {})

    handlers = captured["handlers"]
    assert isinstance(handlers, tuple)
    proxy_handler = next(
        handler for handler in handlers if isinstance(handler, ProxyHandler)
    )
    assert any(isinstance(handler, HTTPSHandler) for handler in handlers)
    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == f"{scheme}://api.example.com:443/quelware.Service/Call"
    assert vars(proxy_handler)["proxies"] == {scheme: proxy_url}
    assert captured["timeout"] == 7.5
    assert response == b"response-body"


@pytest.mark.parametrize(
    "proxy_url",
    [
        "",
        "socks5://user:password@proxy.example.com:1080",
        "http:///missing-host",
        "http://user:password@proxy.example.com:not-a-port",
        "http://user:password@proxy.example.com:3128/path?query=value",
    ],
)
def test_channel_rejects_invalid_proxy_url_without_disclosing_it(
    proxy_url: str,
) -> None:
    """Invalid proxy URLs should fail without exposing credentials in the error."""
    with pytest.raises(ValueError, match="valid HTTP") as exc_info:
        ProtobufHttpChannel(
            "api.example.com",
            443,
            scheme="https",
            proxy_url=proxy_url,
        )

    if proxy_url:
        assert proxy_url not in str(exc_info.value)


def test_channel_rejects_https_proxy_url_without_disclosing_it() -> None:
    """HTTPS proxy URLs should fail without exposing credentials in the error."""
    proxy_url = "https://user:password@proxy.example.com:3128"

    with pytest.raises(ValueError, match="HTTP proxy") as exc_info:
        ProtobufHttpChannel(
            "api.example.com",
            443,
            scheme="https",
            proxy_url=proxy_url,
        )

    assert proxy_url not in str(exc_info.value)


def test_channel_rejects_https_proxy_url_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTTPS proxy URLs from the environment should fail before connecting."""
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example.com:3128")

    with pytest.raises(ValueError, match="HTTP proxy"):
        ProtobufHttpChannel("api.example.com", 443, scheme="https")


def test_channel_rejects_non_protobuf_success_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful non-protobuf response should become a gRPC error."""
    _patch_channel_opener(
        monkeypatch,
        lambda *args, **kwargs: _HttpResponse(b"login", "text/html"),
    )
    channel = ProtobufHttpChannel("api.example.com", 443, scheme="https")

    with pytest.raises(GRPCError, match="unexpected HTTP response content-type"):
        _request(channel, {})


@pytest.mark.parametrize(
    "redirect_url",
    [
        "https://attacker.example.net/steal",
        "http://api.example.com/steal",
        "https://api.example.com:444/steal",
    ],
)
@pytest.mark.parametrize(
    "proxy_url",
    [None, "http://proxy.example.com:3128"],
)
def test_channel_refuses_cross_origin_redirect_before_forwarding_credentials(
    monkeypatch: pytest.MonkeyPatch,
    redirect_url: str,
    proxy_url: str | None,
) -> None:
    """Cross-origin redirects should fail without forwarding request credentials."""
    target_headers: dict[str, str] = {}

    class _RedirectingOpener:
        def __init__(self, handlers: tuple[object, ...]) -> None:
            self._redirect_handler = next(
                (
                    handler
                    for handler in handlers
                    if isinstance(handler, HTTPRedirectHandler)
                ),
                HTTPRedirectHandler(),
            )

        def open(
            self,
            request: Request,
            *,
            timeout: float | None,
        ) -> _HttpResponse:
            del timeout
            redirect_response = _HttpResponse(b"", "text/plain")
            redirected = self._redirect_handler.redirect_request(
                request,
                cast(Any, redirect_response),
                302,
                "Found",
                redirect_response.headers,
                redirect_url,
            )
            assert redirected is not None
            target_headers.update(
                {name.lower(): value for name, value in redirected.header_items()}
            )
            return _HttpResponse(b"response-body", "application/protobuf")

    monkeypatch.setattr(
        transport_module,
        "build_opener",
        lambda *handlers: _RedirectingOpener(handlers),
    )
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme="https",
        proxy_url=proxy_url,
        secret_headers={"x-api-key": "configured-secret"},
    )

    with pytest.raises(GRPCError) as exc_info:
        _request(channel, {"x-pat": "personal-access-token"})

    assert exc_info.value.status is Status.UNKNOWN
    assert target_headers == {}


@pytest.mark.parametrize("redirect_code", [301, 302, 303, 307, 308])
def test_channel_preserves_post_across_same_origin_redirect(
    monkeypatch: pytest.MonkeyPatch,
    redirect_code: int,
) -> None:
    """Same-origin redirects should preserve the protobuf POST request."""
    target_headers: dict[str, str] = {}
    target_request: dict[str, object] = {}

    class _RedirectingOpener:
        def __init__(self, handlers: tuple[object, ...]) -> None:
            self._redirect_handler = next(
                (
                    handler
                    for handler in handlers
                    if isinstance(handler, HTTPRedirectHandler)
                ),
                HTTPRedirectHandler(),
            )
            self._redirect_handler.add_parent(cast(Any, self))
            self._redirected = False

        def open(
            self,
            request: Request,
            *,
            timeout: float | None,
        ) -> _HttpResponse:
            if not self._redirected:
                self._redirected = True
                request.timeout = timeout
                redirect_response = _HttpResponse(b"", "text/plain")
                redirect_response.headers["location"] = (
                    "https://api.example.com/redirected"
                )
                redirect = getattr(
                    self._redirect_handler,
                    f"http_error_{redirect_code}",
                )
                return redirect(
                    request,
                    cast(Any, redirect_response),
                    redirect_code,
                    "Found",
                    redirect_response.headers,
                )
            target_request.update(method=request.get_method(), body=request.data)
            target_headers.update(
                {name.lower(): value for name, value in request.header_items()}
            )
            return _HttpResponse(b"response-body", "application/protobuf")

    monkeypatch.setattr(
        transport_module,
        "build_opener",
        lambda *handlers: _RedirectingOpener(handlers),
    )
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme="https",
        secret_headers={"x-api-key": "configured-secret"},
    )

    response = _request(channel, {"x-pat": "personal-access-token"})

    assert response == b"response-body"
    assert target_request == {"method": "POST", "body": b"request-body"}
    assert target_headers["content-type"] == "application/protobuf"
    assert target_headers["x-api-key"] == "configured-secret"
    assert target_headers["x-pat"] == "personal-access-token"


def test_channel_enforces_default_timeout_across_complete_rpc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default timeout should bound the complete HTTP RPC wall-clock time."""
    release = Event()
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme="https",
        default_timeout_seconds=0.01,
    )

    def _blocking_post(
        route: str,
        body: bytes,
        headers: object,
        timeout: float | None,
    ) -> bytes:
        del route, body, headers, timeout
        release.wait(timeout=0.2)
        return b"response-body"

    monkeypatch.setattr(channel, "_post", _blocking_post)

    async def _invoke() -> None:
        stream = channel.request(
            "/quelware.Service/Call",
            Cardinality.UNARY_UNARY,
            _RequestMessage,
            _ResponseMessage,
        )
        try:
            with pytest.raises(GRPCError) as exc_info:
                await stream.send_message(_RequestMessage(), end=True)
        finally:
            release.set()
        assert exc_info.value.status is Status.DEADLINE_EXCEEDED

    asyncio.run(_invoke())


def test_channel_rechecks_grpclib_deadline_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired grpclib deadline should fail before starting the HTTP RPC."""

    class _Deadline:
        remaining = 1.0

        def time_remaining(self) -> float:
            return self.remaining

    deadline = _Deadline()
    channel = ProtobufHttpChannel("api.example.com", 443, scheme="https")
    monkeypatch.setattr(
        channel,
        "_post",
        lambda *args, **kwargs: pytest.fail("expired request should not be sent"),
    )

    async def _invoke() -> None:
        stream = channel.request(
            "/quelware.Service/Call",
            Cardinality.UNARY_UNARY,
            _RequestMessage,
            _ResponseMessage,
            deadline=deadline,
        )
        deadline.remaining = 0.0
        with pytest.raises(GRPCError) as exc_info:
            await stream.send_message(_RequestMessage(), end=True)
        assert exc_info.value.status is Status.DEADLINE_EXCEEDED

    asyncio.run(_invoke())


def test_channel_maps_socket_timeout_to_deadline_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket timeout should become a deadline-exceeded gRPC error."""

    class _TimeoutOpener:
        def open(self, *args: object, **kwargs: object) -> _HttpResponse:
            raise TimeoutError("timed out")

    monkeypatch.setattr(
        transport_module,
        "build_opener",
        lambda *handlers: _TimeoutOpener(),
    )
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme="https",
        proxy_url="http://proxy.example.com:3128",
    )

    with pytest.raises(GRPCError) as exc_info:
        _request(channel, {})

    assert exc_info.value.status is Status.DEADLINE_EXCEEDED


def test_channel_maps_truncated_response_to_stream_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated response body should become a retryable stream termination."""
    incomplete_read = IncompleteRead(b"partial", 8)

    class _TruncatedResponseOpener:
        def open(self, *args: object, **kwargs: object) -> _HttpResponse:
            return _HttpResponse(
                b"",
                "application/protobuf",
                read_error=incomplete_read,
            )

    monkeypatch.setattr(
        transport_module,
        "build_opener",
        lambda *handlers: _TruncatedResponseOpener(),
    )
    channel = ProtobufHttpChannel(
        "api.example.com",
        443,
        scheme="https",
        proxy_url="http://proxy.example.com:3128",
    )

    with pytest.raises(StreamTerminatedError) as exc_info:
        _request(channel, {})

    assert exc_info.value.__cause__ is incomplete_read
