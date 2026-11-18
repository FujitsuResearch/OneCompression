"""Security, expiry, and concurrency tests for MCP state references."""

import threading
import time

import pytest

from onecomp.mcp.session_registry import SessionRegistry


class Resource:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_token_and_tenant_are_both_required_without_leaking_existence():
    registry = SessionRegistry(max_sessions=2)
    resource = Resource()
    handle = registry.create(
        resource,
        tenant_id="tenant-a",
        require_token=True,
    )
    assert (
        registry.get(
            handle.session_id,
            session_token=handle.session_token,
            tenant_id="tenant-a",
        )
        is resource
    )
    for token, tenant in (
        (None, "tenant-a"),
        ("wrong", "tenant-a"),
        (handle.session_token, "tenant-b"),
    ):
        with pytest.raises(ValueError, match="Unknown session_id"):
            registry.get(
                handle.session_id,
                session_token=token,
                tenant_id=tenant,
            )


def test_expiry_closes_resource_and_releases_capacity(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "onecomp.mcp.session_registry.time.monotonic",
        lambda: clock[0],
    )
    registry = SessionRegistry(max_sessions=1)
    first = Resource()
    handle = registry.create(first, ttl_seconds=10, require_token=False)
    clock[0] = 111.0
    assert registry.cleanup_expired() == 1
    assert first.closed
    with pytest.raises(ValueError, match="Unknown session_id"):
        registry.get(handle.session_id)

    second = Resource()
    registry.create(second, ttl_seconds=10, require_token=False)
    assert len(registry.resources) == 1


def test_access_refreshes_idle_expiry(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "onecomp.mcp.session_registry.time.monotonic",
        lambda: clock[0],
    )
    registry = SessionRegistry()
    resource = Resource()
    handle = registry.create(resource, ttl_seconds=5, require_token=False)
    clock[0] = 14.0
    assert registry.get(handle.session_id) is resource
    clock[0] = 18.0
    assert registry.cleanup_expired() == 0
    clock[0] = 20.0
    assert registry.cleanup_expired() == 1


def test_close_waits_for_active_lease():
    registry = SessionRegistry()
    resource = Resource()
    handle = registry.create(resource, require_token=False)
    close_started = threading.Event()
    close_finished = threading.Event()

    def close_session():
        close_started.set()
        registry.close(handle.session_id)
        close_finished.set()

    with registry.lease(handle.session_id):
        thread = threading.Thread(target=close_session)
        thread.start()
        assert close_started.wait(timeout=1)
        time.sleep(0.02)
        assert not close_finished.is_set()
        assert not resource.closed
    thread.join(timeout=1)
    assert close_finished.is_set()
    assert resource.closed


def test_expiry_cleanup_does_not_hold_global_lock_while_waiting_for_a_lease(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "onecomp.mcp.session_registry.time.monotonic",
        lambda: clock[0],
    )
    registry = SessionRegistry(max_sessions=2)
    first = Resource()
    handle = registry.create(first, ttl_seconds=5, require_token=False)
    cleanup_finished = threading.Event()

    with registry.lease(handle.session_id):
        clock[0] = 106.0
        cleanup = threading.Thread(
            target=lambda: (registry.cleanup_expired(), cleanup_finished.set())
        )
        cleanup.start()
        time.sleep(0.02)
        assert not cleanup_finished.is_set()

        second = Resource()
        registry.create(second, ttl_seconds=5, require_token=False)
        assert len(registry.resources) == 1

    cleanup.join(timeout=1)
    assert cleanup_finished.is_set()
    assert first.closed


def test_waiting_lease_refreshes_expiry_from_actual_acquisition_time(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(
        "onecomp.mcp.session_registry.time.monotonic",
        lambda: clock[0],
    )
    registry = SessionRegistry()
    resource = Resource()
    handle = registry.create(resource, ttl_seconds=5, require_token=False)
    acquired = threading.Event()

    def wait_for_resource():
        with registry.lease(handle.session_id):
            acquired.set()

    with registry.lease(handle.session_id):
        waiter = threading.Thread(target=wait_for_resource)
        waiter.start()
        time.sleep(0.02)
        clock[0] = 100.0
    waiter.join(timeout=1)
    assert acquired.is_set()

    clock[0] = 104.0
    assert registry.cleanup_expired() == 0
