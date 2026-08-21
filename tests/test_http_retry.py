"""Unit tests for HTTP retry classification."""

import aiohttp
import pytest
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from areal.infra.utils.http import HTTPRequestError, arequest_with_retry


class _ErrorResponse:
    content_type = "application/json"

    def __init__(self, status: int) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self) -> None:
        url = URL("http://127.0.0.1:1/generate")
        request_info = aiohttp.RequestInfo(
            url,
            "POST",
            CIMultiDictProxy(CIMultiDict()),
            real_url=url,
        )
        raise aiohttp.ClientResponseError(
            request_info,
            (),
            status=self.status,
            message="test response",
        )


class _StatusSession:
    def __init__(self, status: int) -> None:
        self.status = status
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return _ErrorResponse(self.status)


class _TimeoutResponse:
    async def __aenter__(self):
        raise TimeoutError

    async def __aexit__(self, *_args):
        return False


class _TimeoutSession:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, *_args, **_kwargs):
        self.calls += 1
        return _TimeoutResponse()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [400, 404, 408, 425, 429, 500],
)
async def test_arequest_http_errors_retain_configured_retry_limit(status: int) -> None:
    """Ordinary HTTP requests retain their configured retry behavior."""
    session = _StatusSession(status)

    with pytest.raises(HTTPRequestError, match="after 3 attempts") as caught:
        await arequest_with_retry(
            "127.0.0.1:1",
            "/generate",
            session=session,
            max_retries=3,
            retry_delay=0,
        )

    assert session.calls == 3
    assert caught.value.attempts == 3
    assert caught.value.status == status


@pytest.mark.asyncio
async def test_arequest_supports_a_one_attempt_request() -> None:
    """A caller can make a probe fail after one HTTP attempt."""
    session = _StatusSession(400)

    with pytest.raises(HTTPRequestError, match="after 1 attempt") as caught:
        await arequest_with_retry(
            "127.0.0.1:1",
            "/generate",
            session=session,
            max_retries=1,
            retry_delay=0,
        )

    assert session.calls == 1
    assert caught.value.attempts == 1
    assert caught.value.status == 400


@pytest.mark.asyncio
async def test_arequest_timeout_retries_to_configured_limit() -> None:
    """Transport timeouts retain the configured retry behavior."""
    session = _TimeoutSession()

    with pytest.raises(HTTPRequestError, match="after 3 attempts") as caught:
        await arequest_with_retry(
            "127.0.0.1:1",
            "/generate",
            session=session,
            max_retries=3,
            retry_delay=0,
        )

    assert session.calls == 3
    assert caught.value.attempts == 3
    assert caught.value.status is None
    assert isinstance(caught.value.last_exception, TimeoutError)
