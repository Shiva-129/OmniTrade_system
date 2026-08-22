"""
Shared Phase 0/4 fixtures.

`live_redis`: REAL Redis at localhost:6379 (project-documented config),
isolated on DB 15 and flushed before/after each test. Tests using this
fixture FAIL loudly when Redis is unreachable -- never skipped.
"""
import pytest
import redis

TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture()
def live_redis():
    client = redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except Exception as e:
        pytest.fail(
            f"These tests require a running Redis at localhost:6379 "
            f"(start with: wsl -d kali-linux -e bash -c 'redis-server --daemonize yes'). "
            f"Underlying error: {e}"
        )
    client.flushdb()  # isolated scratch DB for deterministic assertions
    yield client
    client.flushdb()
    client.close()
