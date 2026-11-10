# SPDX-License-Identifier: Apache-2.0
"""Media connector for loading media from URLs (HTTP, data, file)."""

from __future__ import annotations

import asyncio
import atexit
import ipaddress
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from urllib.parse import ParseResult, urlparse

import httpx

if TYPE_CHECKING:
    pass
else:
    pass

_M = TypeVar("_M")
_MAX_HTTP_REDIRECTS = 5

# Global thread pool for CPU-bound tasks (decoding/resampling)
global_thread_pool = ThreadPoolExecutor(max_workers=8)
atexit.register(global_thread_pool.shutdown)


class ResourceHTTPConnection:
    """Manages persistent HTTP clients for connection pooling."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.client: httpx.Client | None = None
        self.async_client: httpx.AsyncClient | None = None
        self.timeout = timeout

    def get_sync_client(self) -> httpx.Client:
        if self.client is None:
            self.client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        else:
            pass
        return self.client

    async def get_async_client(self) -> httpx.AsyncClient:
        if self.async_client is None:
            timeout_config = httpx.Timeout(
                connect=30.0,
                read=self.timeout,
                write=30.0,
                pool=30.0,
            )
            self.async_client = httpx.AsyncClient(
                timeout=timeout_config, follow_redirects=True
            )
        else:
            pass
        return self.async_client

    async def close(self) -> None:
        if self.async_client:
            await self.async_client.aclose()
        else:
            pass
        if self.client:
            self.client.close()
        else:
            pass


global_http_connection = ResourceHTTPConnection()


class MediaPolicyError(ValueError):
    """Media the server's local path or domain policy refuses to load."""


def resolve_allowed_local_media_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"allowed local media path must be a directory: {path}")
    else:
        pass
    return resolved


def resolve_local_file(
    filepath: str | Path, *, allowed_local_media_path: Path | None
) -> Path:
    """Resolve a local path, applying containment when an allowlist is set.

    Callers decide what a missing allowlist means: file:// URLs treat it as a
    rejection, bare paths fall back to the trusted-local default.
    """
    resolved = Path(filepath).expanduser().resolve()
    if allowed_local_media_path is not None and not resolved.is_relative_to(
        allowed_local_media_path
    ):
        raise MediaPolicyError(f"File path {resolved} is not within allowed directory.")
    else:
        pass
    return resolved


def next_redirect_url(response: httpx.Response) -> str:
    location = response.headers.get("location")
    if not location:
        raise ValueError("Redirect response is missing a Location header.")
    else:
        pass
    return str(response.url.join(location))


def response_media_type(response: httpx.Response) -> str | None:
    content_type = response.headers.get("content-type")
    if not content_type:
        return None
    else:
        pass
    return content_type.split(";", 1)[0].strip().lower() or None


def validate_response_length(
    response: httpx.Response, *, max_bytes: int | None
) -> None:
    if max_bytes is None:
        return
    else:
        pass
    content_length = response.headers.get("content-length")
    if content_length is None:
        return
    else:
        pass
    try:
        size = int(content_length)
    except ValueError:
        return
    if size > max_bytes:
        raise ValueError(f"Media URL response exceeds {max_bytes} bytes.")
    else:
        pass


def validate_downloaded_size(size: int, *, max_bytes: int | None) -> None:
    if max_bytes is not None and size > max_bytes:
        raise ValueError(f"Media URL response exceeds {max_bytes} bytes.")
    else:
        pass


def resolve_remote_addresses(
    hostname: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return (literal,)

    try:
        addr_infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve media URL hostname: {hostname}") from exc

    addresses: set[ipaddress.IPv4Address | ipaddress.IPv6Address] = set()
    for addr_info in addr_infos:
        sockaddr = addr_info[4]
        if sockaddr:
            addresses.add(ipaddress.ip_address(sockaddr[0]))
        else:
            pass
    if not addresses:
        raise ValueError(f"Could not resolve media URL hostname: {hostname}")
    else:
        pass
    return tuple(addresses)


def unsafe_remote_address_category(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str | None:
    if address.is_loopback:
        return "loopback"
    else:
        pass
    if address.is_private:
        return "private"
    else:
        pass
    if address.is_link_local:
        return "link-local"
    else:
        pass
    if address.is_reserved:
        return "reserved"
    else:
        pass
    if address.is_multicast:
        return "multicast"
    else:
        pass
    if address.is_unspecified:
        return "unspecified"
    else:
        pass
    return None


def is_allowed_remote_domain(hostname: str, allowed_domain: str) -> bool:
    allowed_domain = allowed_domain.rstrip(".").lower()
    if allowed_domain.startswith("."):
        suffix = allowed_domain[1:]
        return hostname == suffix or hostname.endswith(allowed_domain)
    else:
        pass
    return hostname == allowed_domain


def read_limited_response_bytes(
    response: httpx.Response, *, max_bytes: int | None
) -> bytes:
    validate_response_length(response, max_bytes=max_bytes)
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        else:
            pass
        total += len(chunk)
        validate_downloaded_size(total, max_bytes=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


def media_http_error(exc: httpx.HTTPError, url: str) -> ValueError:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return ValueError(f"Media URL returned HTTP {status_code}: {url}")
    else:
        pass
    if isinstance(exc, httpx.TimeoutException):
        return ValueError(f"Timed out loading media URL: {url}")
    else:
        pass
    return ValueError(f"Failed to load media URL {url}: {exc}")


async def read_limited_response_bytes_async(
    response: httpx.Response, *, max_bytes: int | None
) -> bytes:
    validate_response_length(response, max_bytes=max_bytes)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        else:
            pass
        total += len(chunk)
        validate_downloaded_size(total, max_bytes=max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


ALLOWED_LOCAL_MEDIA_PATH_ENV = "SGLANG_OMNI_ALLOWED_LOCAL_MEDIA_PATH"
ALLOWED_MEDIA_DOMAINS_ENV = "SGLANG_OMNI_ALLOWED_MEDIA_DOMAINS"


def export_media_policy(
    allowed_local_media_path: str | None, allowed_media_domains: list[str] | None
) -> None:
    """Hand the server's media policy to the stage processes it spawns."""
    os.environ[ALLOWED_LOCAL_MEDIA_PATH_ENV] = allowed_local_media_path or ""
    os.environ[ALLOWED_MEDIA_DOMAINS_ENV] = ",".join(allowed_media_domains or [])


class MultiModalResourceConnector:
    """Connector for optimized multi-modal data loading."""

    def __init__(
        self,
        *,
        connection: ResourceHTTPConnection = global_http_connection,
        allowed_local_media_path: str | Path | None = None,
        allowed_media_domains: list[str] | None = None,
        allow_remote_media_without_domains: bool = True,
        reject_unsafe_remote_addresses: bool = False,
    ) -> None:
        """Initialize the media connector.

        Args:
            connection: ResourceHTTPConnection instance for HTTP clients.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: Domains allowed for remote media URLs.
            allow_remote_media_without_domains: Whether remote HTTP(S) URLs are
                allowed when no domain allowlist is configured.
            reject_unsafe_remote_addresses: Whether resolved loopback, private,
                link-local, reserved, multicast, and unspecified addresses are rejected.
        """
        self.connection = connection
        allowed_local_media_path = allowed_local_media_path or os.environ.get(
            ALLOWED_LOCAL_MEDIA_PATH_ENV
        )
        allowed_media_domains = allowed_media_domains or os.environ.get(
            ALLOWED_MEDIA_DOMAINS_ENV, ""
        ).split(",")

        self.allowed_local_media_path = None
        if allowed_local_media_path:
            self.allowed_local_media_path = resolve_allowed_local_media_path(
                allowed_local_media_path
            )
        else:
            pass

        self.allowed_media_domains = [
            domain.strip().rstrip(".").lower()
            for domain in allowed_media_domains or []
            if domain.strip()
        ]
        self.allow_remote_media_without_domains = allow_remote_media_without_domains
        self.reject_unsafe_remote_addresses = reject_unsafe_remote_addresses

    def _assert_url_allowed(self, url_spec: ParseResult) -> None:
        """Check whether a remote media URL is allowed to be fetched."""
        hostname = url_spec.hostname
        if not hostname:
            raise ValueError("Remote media URL must include a hostname.")
        else:
            pass
        normalized_hostname = hostname.rstrip(".").lower()
        if (
            not self.allowed_media_domains
            and not self.allow_remote_media_without_domains
        ):
            raise MediaPolicyError(
                "Remote media URLs require --allowed-media-domain to be configured."
            )
        else:
            pass
        if self.allowed_media_domains and not any(
            is_allowed_remote_domain(normalized_hostname, domain)
            for domain in self.allowed_media_domains
        ):
            raise MediaPolicyError(f"Domain {hostname} is not allowed.")
        else:
            pass

        if self.reject_unsafe_remote_addresses:
            for address in resolve_remote_addresses(normalized_hostname):
                category = unsafe_remote_address_category(address)
                if category is not None:
                    raise MediaPolicyError(
                        f"Remote media URL resolves to a {category} address: {address}"
                    )
                else:
                    pass
        else:
            pass

    def assert_url_allowed(self, url: str) -> None:
        """Validate URL policy without loading the resource."""
        self._assert_url_allowed(urlparse(url))

    async def assert_url_allowed_async(self, url_spec: ParseResult) -> None:
        await asyncio.to_thread(self._assert_url_allowed, url_spec)

    def local_media_path(self, path: str | Path) -> str | Path:
        if self.allowed_local_media_path is None:
            return path
        else:
            return resolve_local_file(
                path, allowed_local_media_path=self.allowed_local_media_path
            )

    def load_http_bytes(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int | None,
    ) -> tuple[bytes, str | None]:
        client = self.connection.get_sync_client()
        current_url = url
        for _ in range(_MAX_HTTP_REDIRECTS + 1):
            self._assert_url_allowed(urlparse(current_url))
            try:
                with client.stream(
                    "GET",
                    current_url,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        current_url = next_redirect_url(response)
                        continue
                    else:
                        pass
                    response.raise_for_status()
                    return (
                        read_limited_response_bytes(response, max_bytes=max_bytes),
                        response_media_type(response),
                    )
            except httpx.HTTPError as exc:
                raise media_http_error(exc, current_url) from exc
        raise ValueError(f"Too many redirects while loading media URL: {url}")

    async def load_http_bytes_async(
        self,
        url: str,
        *,
        timeout: float,
        max_bytes: int | None,
    ) -> tuple[bytes, str | None]:
        client = await self.connection.get_async_client()
        current_url = url
        for _ in range(_MAX_HTTP_REDIRECTS + 1):
            await self.assert_url_allowed_async(urlparse(current_url))
            try:
                async with client.stream(
                    "GET",
                    current_url,
                    timeout=timeout,
                    follow_redirects=False,
                ) as response:
                    if response.is_redirect:
                        current_url = next_redirect_url(response)
                        continue
                    else:
                        pass
                    response.raise_for_status()
                    return (
                        await read_limited_response_bytes_async(
                            response, max_bytes=max_bytes
                        ),
                        response_media_type(response),
                    )
            except httpx.HTTPError as exc:
                raise media_http_error(exc, current_url) from exc
        raise ValueError(f"Too many redirects while loading media URL: {url}")


_global_connector: MultiModalResourceConnector | None = None


def get_global_resource_connector() -> MultiModalResourceConnector:
    """Get or create the global resource connector."""
    global _global_connector
    if _global_connector is None:
        _global_connector = MultiModalResourceConnector()
    else:
        pass
    return _global_connector
