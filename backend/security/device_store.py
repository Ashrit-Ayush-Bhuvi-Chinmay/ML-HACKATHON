
import sqlite3
import hashlib
import secrets

from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "auth.db"

SESSION_IDLE_MINUTES = 30
SESSION_MAX_HOURS = 4


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                device_name TEXT NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked INTEGER DEFAULT 0,
                role TEXT NOT NULL DEFAULT 'member'
            )
        """)

        # Migrate databases created before role support
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(devices)"
            ).fetchall()
        }

        if "role" not in columns:
            conn.execute("""
                ALTER TABLE devices
                ADD COLUMN role TEXT NOT NULL DEFAULT 'member'
            """)


def hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


def register_device(device_name):
    device_id = str(uuid4())
    access_token = secrets.token_urlsafe(32)

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_MAX_HOURS)

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO devices (
                device_id,
                device_name,
                token_hash,
                created_at,
                last_seen,
                expires_at,
                revoked,
                role
            )
            VALUES (?, ?, ?, ?, ?, ?, 0, 'member')
        """, (
            device_id,
            device_name,
            hash_token(access_token),
            now.isoformat(),
            now.isoformat(),
            expires_at.isoformat()
        ))

    return {
        "device_id": device_id,
        "access_token": access_token,
        "expires_at": expires_at.isoformat()
    }


def get_authenticated_device(token):
    now = datetime.now(timezone.utc)
    token_digest = hash_token(token)

    with get_connection() as conn:
        device = conn.execute("""
            SELECT *
            FROM devices
            WHERE token_hash = ?
            AND revoked = 0
        """, (token_digest,)).fetchone()

        if not device:
            return None

        expires_at = datetime.fromisoformat(
            device["expires_at"]
        )

        last_seen = datetime.fromisoformat(
            device["last_seen"]
        )

        if now >= expires_at:
            return None

        if now - last_seen > timedelta(
            minutes=SESSION_IDLE_MINUTES
        ):
            return None

        conn.execute("""
            UPDATE devices
            SET last_seen = ?
            WHERE device_id = ?
        """, (
            now.isoformat(),
            device["device_id"]
        ))

        return {
            "device_id": device["device_id"],
            "device_name": device["device_name"],
            "role": device["role"]
        }


def list_devices():
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                device_id,
                device_name,
                created_at,
                last_seen,
                expires_at,
                revoked,
                role
            FROM devices
            ORDER BY created_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


def revoke_device(device_id):
    with get_connection() as conn:
        result = conn.execute("""
            UPDATE devices
            SET revoked = 1
            WHERE device_id = ?
            AND revoked = 0
        """, (device_id,))

        return result.rowcount > 0


def promote_to_admin(device_id):
    with get_connection() as conn:
        result = conn.execute("""
            UPDATE devices
            SET role = 'admin'
            WHERE device_id = ?
            AND revoked = 0
        """, (device_id,))

        return result.rowcount > 0
