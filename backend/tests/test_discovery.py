"""UDP discovery announcement checks, moved from the old print-based test_discovery.py."""
import pytest

from config import settings
from networking import discovery_security as ds

SECRET = "ab" * 32  # 32 bytes as hex


@pytest.fixture(autouse=True)
def discovery_secret(monkeypatch):
    monkeypatch.setattr(settings, "DISCOVERY_SECRET", SECRET)
    ds._seen_nonces.clear()
    yield
    ds._seen_nonces.clear()


def _resign(announcement):
    unsigned = {key: value for key, value in announcement.items() if key != "signature"}
    announcement["signature"] = ds.sign_announcement(unsigned)
    return announcement


def test_authentic_announcement_is_accepted():
    announcement = ds.create_announcement("192.168.1.5", 8000)
    assert announcement["scheme"] == "http"
    assert ds.verify_announcement(announcement) is True


def test_replayed_announcement_is_rejected():
    announcement = ds.create_announcement("192.168.1.5", 8000)
    assert ds.verify_announcement(announcement) is True
    assert ds.verify_announcement(announcement) is False


def test_tampered_announcement_is_rejected():
    announcement = ds.create_announcement("192.168.1.5", 8000)
    announcement["host"] = "192.168.1.100"
    assert ds.verify_announcement(announcement) is False


def test_https_announcement_is_rejected():
    announcement = ds.create_announcement("192.168.1.5", 8000)
    announcement["scheme"] = "https"
    _resign(announcement)
    assert ds.verify_announcement(announcement) is False


def test_missing_secret_refuses_to_start(monkeypatch):
    monkeypatch.setattr(settings, "DISCOVERY_SECRET", "")
    with pytest.raises(RuntimeError):
        ds.get_discovery_key()


def test_malformed_secret_refuses_to_start(monkeypatch):
    monkeypatch.setattr(settings, "DISCOVERY_SECRET", "not-hex")
    with pytest.raises(RuntimeError):
        ds.get_discovery_key()
