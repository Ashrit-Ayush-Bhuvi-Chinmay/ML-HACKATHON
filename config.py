"""Runtime settings for the PathSnap API.

Every value can be overridden with an environment variable of the same name or a `.env`
file next to this module. See `.env.example`.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

#: Folder scanned for trips when PATHSNAP_EVAL_DIR is not set.
DEFAULT_EVAL_SUBPATH = Path("Provided Material") / "Student folder" / "student_test"


def find_default_eval_dir() -> Path | None:
    """Walk up from this checkout to the ML_HACKATHON root and return the student_test folder.

    The API lives in ``branches/secure-backend``; the supplied trips live in
    ``Provided Material/Student folder/student_test``. Searching upwards keeps the default
    working from any checkout depth (worktree, clone, zip extract).
    """
    for parent in BASE_DIR.parents:
        candidate = parent / DEFAULT_EVAL_SUBPATH
        if candidate.is_dir():
            return candidate
    return None


class Settings(BaseSettings):

    APP_NAME: str = "PathSnap Backend"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # "*" lets laptop 2's UI (unknown origin) call the API. Credentials are not used.
    ALLOWED_ORIGINS: str = "*"

    # Request rate limits (slowapi). Kept from the original security layer.
    RATE_LIMIT: str = "60/minute"

    # Folder containing one sub-folder per trip, or all trip files flat. Empty = auto-detect.
    PATHSNAP_EVAL_DIR: str = ""

    # Optional device pairing / bearer tokens. Off for the demo: the UI sends no token.
    PATHSNAP_REQUIRE_AUTH: bool = False

    # Optional UDP LAN discovery. Requires a 64-hex-character DISCOVERY_SECRET.
    PATHSNAP_DISCOVERY: bool = False
    DISCOVERY_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ] or ["*"]

    @property
    def eval_dir(self) -> Path:
        """Trip folder from PATHSNAP_EVAL_DIR, or the supplied student_test folder."""
        if self.PATHSNAP_EVAL_DIR.strip():
            path = Path(self.PATHSNAP_EVAL_DIR.strip()).expanduser()
            if not path.is_absolute():
                path = (BASE_DIR / path).resolve()
            return path

        default = find_default_eval_dir()
        if default is None:
            raise RuntimeError(
                "PATHSNAP_EVAL_DIR is not set and the default student_test folder could not be "
                "found. Point PATHSNAP_EVAL_DIR at the folder containing the trip sub-folders."
            )
        return default


settings = Settings()
