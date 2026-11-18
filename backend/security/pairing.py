
import secrets
import hashlib
import hmac
import time
import threading

from config import settings


class PairingManager:
    def __init__(self):
        self.code_hash = None
        self.expires_at = None
        self.failed_attempts = 0
        self.max_attempts = 5
        self.lock = threading.Lock()

    def generate_code(self):
        # Avoid ambiguous characters such as 0/O and 1/I
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        code = "".join(
            secrets.choice(alphabet) for _ in range(12)
        )

        digest = hashlib.sha256(code.encode()).hexdigest()

        with self.lock:
            self.code_hash = digest
            self.expires_at = time.time() + 120
            self.failed_attempts = 0

        return code

    def verify_code(self, submitted_code):
        if not submitted_code:
            return False

        digest = hashlib.sha256(
            submitted_code.strip().upper().encode()
        ).hexdigest()

        with self.lock:
            if not self.code_hash:
                return False

            if time.time() > self.expires_at:
                self._invalidate()
                return False

            if self.failed_attempts >= self.max_attempts:
                self._invalidate()
                return False

            if not hmac.compare_digest(digest, self.code_hash):
                self.failed_attempts += 1

                if self.failed_attempts >= self.max_attempts:
                    self._invalidate()

                return False

            # One-time use: immediately invalidate after success
            self._invalidate()
            return True

    def _invalidate(self):
        self.code_hash = None
        self.expires_at = None
        self.failed_attempts = 0


pairing_manager = PairingManager()
