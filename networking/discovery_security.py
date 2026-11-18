
import hashlib
import hmac
import json
import secrets
import time
import threading

from config import settings


NONCE_EXPIRY_SECONDS = 60
CLOCK_TOLERANCE_SECONDS = 30

_seen_nonces = {}
_nonce_lock = threading.Lock()


def get_discovery_key():
    secret = settings.DISCOVERY_SECRET.strip()

    try:
        key = bytes.fromhex(secret)
    except ValueError:
        raise RuntimeError(
            "DISCOVERY_SECRET must be a hexadecimal secret"
        )

    if len(key) != 32:
        raise RuntimeError(
            "DISCOVERY_SECRET must contain exactly 32 bytes"
        )

    return key


def canonical_json(data):
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":")
    ).encode("utf-8")


def sign_announcement(data):
    key = get_discovery_key()

    signature = hmac.new(
        key,
        canonical_json(data),
        hashlib.sha256
    ).hexdigest()

    return signature


def create_announcement(host, port):
    announcement = {
        "service": "PathSnap",
        "version": 1,
        "scheme": "http",
        "host": host,
        "port": port,
        "timestamp": int(time.time()),
        "nonce": secrets.token_hex(16)
    }

    announcement["signature"] = sign_announcement(
        announcement
    )

    return announcement


def verify_announcement(data):
    if not isinstance(data, dict):
        return False

    required = {
        "service", "version", "scheme", "host",
        "port", "timestamp", "nonce", "signature"
    }

    if set(data.keys()) != required:
        return False

    if data["service"] != "PathSnap":
        return False

    if data["version"] != 1:
        return False

    if data["scheme"] != "http":
        return False

    if not isinstance(data["host"], str):
        return False

    if not isinstance(data["port"], int):
        return False

    if not 1 <= data["port"] <= 65535:
        return False

    if not isinstance(data["timestamp"], int):
        return False

    if not isinstance(data["nonce"], str):
        return False

    if not isinstance(data["signature"], str):
        return False

    if len(data["nonce"]) != 32:
        return False

    if len(data["signature"]) != 64:
        return False

    # Reject announcements outside the allowed clock window
    if abs(time.time() - data["timestamp"]) > CLOCK_TOLERANCE_SECONDS:
        return False

    unsigned_data = {
        key: value
        for key, value in data.items()
        if key != "signature"
    }

    expected_signature = sign_announcement(unsigned_data)

    if not hmac.compare_digest(
        expected_signature,
        data["signature"]
    ):
        return False

    # Prevent replay of previously accepted announcements
    now = time.time()

    with _nonce_lock:
        expired = [
            nonce for nonce, expiry in _seen_nonces.items()
            if expiry < now
        ]

        for nonce in expired:
            del _seen_nonces[nonce]

        if data["nonce"] in _seen_nonces:
            return False

        _seen_nonces[data["nonce"]] = (
            now + NONCE_EXPIRY_SECONDS
        )

    return True

