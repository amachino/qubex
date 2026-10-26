"""Provide a grpclib-compatible HTTP(S) channel for raw protobuf bodies."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Collection, Mapping
from http.client import IncompleteRead
from types import TracebackType
from typing import Any, cast
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    OpenerDirector,
    ProxyHandler,
    Request,
    build_opener,
    getproxies,
)

from google.protobuf.message import DecodeError
from google.rpc.error_details_pb2 import ErrorInfo
from google.rpc.status_pb2 import Status as RpcStatus
from grpclib.const import Cardinality, Status
from grpclib.exceptions import GRPCError, StreamTerminatedError

from .quelware_transport_config import HttpScheme

_CONTENT_TYPE = "application/protobuf"
_USER_AGENT = "qubex-protobuf-http/1"
_DEFAULT_PORTS = {"http": 80, "https": 443}

_PostRequest = Callable[
    [str, bytes, Mapping[str, str], float | None],
    bytes,
]

_HTTP_STATUS_TO_GRPC_STATUS = {
    400: Status.INVALID_ARGUMENT,
    401: Status.UNAUTHENTICATED,
    403: Status.PERMISSION_DENIED,
    404: Status.NOT_FOUND,
    408: Status.DEADLINE_EXCEEDED,
    409: Status.ABORTED,
    412: Status.FAILED_PRECONDITION,
    429: Status.RESOURCE_EXHAUSTED,
    499: Status.CANCELLED,
    500: Status.INTERNAL,
    501: Status.UNIMPLEMENTED,
    502: Status.UNAVAILABLE,
    503: Status.UNAVAILABLE,
    504: Status.DEADLINE_EXCEEDED,
}


def _validate_proxy_url(proxy_url: str) -> None:
    try:
        parsed = urlsplit(proxy_url)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("proxy URL must be a valid HTTP proxy URL") from exc
    if (
        parsed.scheme.lower() != "http"
        or hostname is None
        or parsed.path not in {"", "/"}
        or bool(parsed.query)
        or bool(parsed.fragment)
    ):
        raise ValueError("proxy URL must be a valid HTTP proxy URL")


def _url_origin(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    default_port = _DEFAULT_PORTS.get(scheme)
    if hostname is None or default_port is None:
        return None
    return scheme, hostname.lower(), default_port if port is None else port


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    http_error_308 = HTTPRedirectHandler.http_error_302

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        """Reject redirects that would forward request headers across origins."""
        if _url_origin(req.full_url) != _url_origin(newurl):
            raise HTTPError(
                req.full_url,
                code,
                "cross-origin redirect refused",
                headers,
                fp,
            )
        return Request(  # noqa: S310 - newurl passed the same-origin check above.
            newurl,
            data=req.data,
            headers=dict(req.headers),
            origin_req_host=req.origin_req_host,
            unverifiable=True,
            method=req.get_method(),
        )


def _metadata_headers(
    metadata: Mapping[str, str | bytes] | Collection[tuple[str, str | bytes]] | None,
) -> dict[str, str]:
    if metadata is None:
        items: Collection[tuple[str, str | bytes]] = ()
    elif isinstance(metadata, Mapping):
        items = cast(Collection[tuple[str, str | bytes]], metadata.items())
    else:
        items = metadata

    headers: dict[str, str] = {}
    for name, value in items:
        normalized_name = name.lower()
        if isinstance(value, bytes):
            if normalized_name.endswith("-bin"):
                normalized_value = base64.b64encode(value).decode("ascii")
            else:
                normalized_value = value.decode("ascii")
        else:
            normalized_value = value
        headers[normalized_name] = normalized_value

    # Transport headers must not be overridden by RPC metadata.
    headers["content-type"] = _CONTENT_TYPE
    headers["accept"] = _CONTENT_TYPE
    headers["user-agent"] = _USER_AGENT
    return headers


def _details_from_rpc_status(rpc_status: RpcStatus) -> list[Any]:
    details: list[Any] = []
    for packed in rpc_status.details:
        if packed.Is(ErrorInfo.DESCRIPTOR):
            error_info = ErrorInfo()
            packed.Unpack(error_info)
            details.append(error_info)
        else:
            details.append(packed)
    return details


def _grpc_error_from_http_error(error: HTTPError, body: bytes) -> GRPCError:
    fallback_status = _HTTP_STATUS_TO_GRPC_STATUS.get(error.code, Status.UNKNOWN)
    fallback_message = f"HTTP {error.code}: {error.reason}"

    if body:
        try:
            rpc_status = RpcStatus.FromString(body)
            status = Status(rpc_status.code)
        except (DecodeError, ValueError, TypeError):
            pass
        else:
            if status is not Status.OK:
                return GRPCError(
                    status,
                    rpc_status.message or fallback_message,
                    _details_from_rpc_status(rpc_status),
                )

    return GRPCError(fallback_status, fallback_message)


def _effective_timeout(timeout: float | None, deadline: Any | None) -> float | None:
    deadline_timeout = None if deadline is None else deadline.time_remaining()
    if timeout is None:
        return deadline_timeout
    if deadline_timeout is None:
        return timeout
    return min(timeout, deadline_timeout)


class _UnaryUnaryHttpStream:
    def __init__(
        self,
        post: _PostRequest,
        route: str,
        response_type: type[Any],
        *,
        timeout: float | None,
        deadline: Any | None,
        metadata: (
            Mapping[str, str | bytes] | Collection[tuple[str, str | bytes]] | None
        ),
    ) -> None:
        self._post = post
        self._route = route
        self._response_type = response_type
        self._timeout = timeout
        self._deadline = deadline
        self._headers = _metadata_headers(metadata)
        self._response_body: bytes | None = None

    async def __aenter__(self) -> _UnaryUnaryHttpStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    async def send_message(self, message: Any, *, end: bool = False) -> None:
        if not end:
            raise NotImplementedError("HTTP(S) transport supports unary RPCs only")
        if self._response_body is not None:
            raise RuntimeError("request message has already been sent")
        effective_timeout = _effective_timeout(self._timeout, self._deadline)
        if effective_timeout is not None and effective_timeout <= 0:
            raise GRPCError(Status.DEADLINE_EXCEEDED, "request deadline exceeded")

        post = asyncio.to_thread(
            self._post,
            self._route,
            bytes(message),
            self._headers,
            effective_timeout,
        )
        try:
            if effective_timeout is None:
                self._response_body = await post
            else:
                self._response_body = await asyncio.wait_for(
                    post,
                    timeout=effective_timeout,
                )
        except asyncio.TimeoutError as exc:
            raise GRPCError(
                Status.DEADLINE_EXCEEDED,
                "request deadline exceeded",
            ) from exc

    async def recv_message(self) -> Any:
        if self._response_body is None:
            raise RuntimeError("request message has not been sent")
        try:
            return self._response_type.parse(self._response_body)
        except Exception as exc:
            raise GRPCError(
                Status.INTERNAL,
                f"invalid protobuf response for {self._route}: {exc}",
            ) from exc


class ProtobufHttpChannel:
    """A minimal grpclib-compatible channel backed by HTTP(S) POST requests."""

    def __init__(
        self,
        host: str,
        port: int | None,
        *,
        scheme: HttpScheme = "https",
        base_path: str = "",
        default_timeout_seconds: float | None = None,
        proxy_url: str | None = None,
        secret_headers: Mapping[str, str] | None = None,
    ) -> None:
        if scheme not in {"http", "https"}:
            raise ValueError("scheme must be http or https")
        if "://" in host or "/" in host:
            raise ValueError("host must not contain a URL scheme or path")
        if not host:
            raise ValueError("host must not be empty")
        if secret_headers and scheme != "https":
            raise ValueError("Secret headers can be used only with HTTPS")
        formatted_host = f"[{host}]" if ":" in host else host
        formatted_port = "" if port is None else f":{port}"
        stripped_base_path = base_path.strip("/")
        normalized_base_path = f"/{stripped_base_path}" if stripped_base_path else ""
        self._origin = (
            f"{scheme}://{formatted_host}{formatted_port}{normalized_base_path}"
        )
        self._default_timeout_seconds = default_timeout_seconds
        opener_handlers: list[Any] = [_SameOriginRedirectHandler()]
        proxies = getproxies() if proxy_url is None else {scheme: proxy_url}
        configured_proxy_url = proxies.get(scheme)
        if configured_proxy_url is not None:
            _validate_proxy_url(configured_proxy_url)
        proxy_handler = ProxyHandler(proxies)
        opener_handlers.extend((proxy_handler, HTTPSHandler()))
        self._opener: OpenerDirector = build_opener(*opener_handlers)
        self._secret_headers = dict(secret_headers or {})

    def request(
        self,
        route: str,
        cardinality: Cardinality,
        request_type: type[Any],
        response_type: type[Any],
        *,
        timeout: float | None = None,
        deadline: Any | None = None,
        metadata: (
            Mapping[str, str | bytes] | Collection[tuple[str, str | bytes]] | None
        ) = None,
    ) -> _UnaryUnaryHttpStream:
        """Create a unary-unary HTTP(S) request stream compatible with grpclib."""
        del request_type
        if cardinality is not Cardinality.UNARY_UNARY:
            raise NotImplementedError(
                "HTTP(S) transport supports unary-unary RPCs only"
            )
        effective_timeout = (
            self._default_timeout_seconds if timeout is None else timeout
        )
        return _UnaryUnaryHttpStream(
            self._post,
            route,
            response_type,
            timeout=effective_timeout,
            deadline=deadline,
            metadata=metadata,
        )

    def close(self) -> None:
        """Match grpclib.Channel.close(); urllib keeps no persistent session."""

    def _post(
        self,
        route: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float | None,
    ) -> bytes:
        # The constructor permits only HTTP(S), and the route is appended as a
        # path, so other URL schemes cannot reach urllib.
        request_headers = dict(headers)
        request_headers.update(self._secret_headers)
        request = Request(  # noqa: S310
            f"{self._origin}/{route.lstrip('/')}",
            data=body,
            headers=request_headers,
            method="POST",
        )
        try:
            response = self._opener.open(
                request,
                timeout=timeout,
            )
            with response:
                content_type = response.headers.get_content_type().lower()
                if content_type != _CONTENT_TYPE:
                    raise GRPCError(
                        Status.UNKNOWN,
                        "unexpected HTTP response content-type "
                        f"for {route}: {content_type!r}",
                    )
                return _read_response_body(response)
        except HTTPError as error:
            try:
                error_body = _read_response_body(error)
            finally:
                error.close()
            raise _grpc_error_from_http_error(error, error_body) from error
        except TimeoutError as exc:
            raise GRPCError(
                Status.DEADLINE_EXCEEDED,
                "request deadline exceeded",
            ) from exc
        except OSError as exc:
            # Existing quelware-client retry handling recognizes this exception
            # and retries only calls which the agent marks as idempotent.
            raise StreamTerminatedError(f"Connection lost: {exc}") from exc


def _read_response_body(response: Any) -> bytes:
    try:
        return cast(bytes, response.read())
    except IncompleteRead as exc:
        raise StreamTerminatedError(f"Connection lost: {exc}") from exc


__all__ = ["ProtobufHttpChannel"]
