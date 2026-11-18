"""Bridge between the HTTP layer and the PathSnap ML package.

Only this module imports ``pathsnap``. It owns the trip registry, the single-prediction lock,
the thread-pool calls and the startup warm-up, so the routes stay thin.

``predict_trip`` reads ``.track``/``.nodes``/``.arcs`` only. ``score_trip`` is the only reader
of ``.route`` and runs server-side, after prediction; neither the reference arcs nor their
geometry are ever returned.
"""
from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from pathsnap.io import find_file, list_trips
from pathsnap.pipeline import (
    experiment_history,
    list_models,
    load_bundle,
    predict_trip,
    score_trip,
)

from config import settings
from utils.logger import logger


def _count_track_points(trip_dir: Path) -> int:
    """Number of GPS observations, counted from the .track file (no pipeline run)."""
    track = find_file(trip_dir, ".track")
    with open(track, encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _fmt(value) -> str:
    if isinstance(value, bool) or value is None:
        return "n/a" if value is None else str(value)
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if math.isfinite(value) else "n/a"
    return "n/a"


def _fmt_ms(value) -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return str(int(round(value)))
    return "?"


class MLAdapter:

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._trips: dict[str, Path] = {}
        self._n_points: dict[str, int] = {}
        self._eval_dir: Path | None = None
        self._model_name: str | None = None

    # ------------------------------------------------------------------ registry

    @property
    def eval_dir(self) -> Path:
        if self._eval_dir is None:
            self._eval_dir = settings.eval_dir
        return self._eval_dir

    @property
    def n_trips(self) -> int:
        return len(self._trips)

    @property
    def model_name(self) -> str:
        if self._model_name is None:
            self._model_name = str(load_bundle()["model_name"])
        return self._model_name

    def has_trip(self, trip_id: str) -> bool:
        return trip_id in self._trips

    def load_trips(self) -> int:
        """Scan the evaluation folder once. Trip ids are folder/file stems, never hardcoded."""
        folder = self.eval_dir
        if not folder.is_dir():
            raise RuntimeError(f"trip folder does not exist: {folder}")

        trips = list_trips(folder)
        if not trips:
            raise RuntimeError(f"no trips found in {folder} (expected *.track files)")

        self._trips = {path.name: path for path in trips}
        self._n_points = {name: _count_track_points(path) for name, path in self._trips.items()}
        logger.info("Loaded %d trips from %s", len(self._trips), folder)
        return len(self._trips)

    def health(self) -> dict:
        return {"status": "ok", "model": self.model_name, "n_trips": self.n_trips}

    def trips(self) -> list[dict]:
        return [
            {"trip_id": trip_id, "n_points": self._n_points[trip_id]}
            for trip_id in sorted(self._trips)
        ]

    # ------------------------------------------------------------------ startup

    async def startup(self) -> None:
        """Load the trip registry, the model bundle and run one prediction to warm the caches."""
        self.load_trips()
        load_started = time.perf_counter()
        await run_in_threadpool(load_bundle)
        logger.info("Model bundle loaded in %.0f ms", (time.perf_counter() - load_started) * 1000)

        if not self._trips:
            return
        first = sorted(self._trips)[0]
        started = time.perf_counter()
        result = await run_in_threadpool(self._run_prediction, first, None, True, "-", False)
        logger.info(
            "WARM-UP trip=%s points=%s total=%sms",
            first,
            result.get("n_points", "?"),
            _fmt_ms(result.get("timings_ms", {}).get("total")),
        )

    # ------------------------------------------------------------------ prediction

    async def predict(self, trip_id: str, model: str | None, use_hmm: bool, client: str) -> dict:
        # One prediction at a time: the pipeline is CPU-bound and each call uses ~1 core.
        async with self._lock:
            return await run_in_threadpool(self._run_prediction, trip_id, model, use_hmm, client, True)

    def _run_prediction(
        self,
        trip_id: str,
        model: str | None,
        use_hmm: bool,
        client: str,
        log: bool,
    ) -> dict:
        trip_dir = self._trips[trip_id]

        prediction = predict_trip(trip_dir, model=model, use_hmm=use_hmm)

        score_started = time.perf_counter()
        metrics = score_trip(trip_dir, prediction)
        score_ms = (time.perf_counter() - score_started) * 1000

        # Keep the reference route server-side: score_trip reports a median distance computed
        # against it, but nothing named "ref*" may appear in the response (leak test).
        metrics = {key: value for key, value in metrics.items() if not key.lower().startswith("ref")}

        if log:
            total_ms = prediction.get("timings_ms", {}).get("total")
            logger.info(
                "PREDICT trip=%s model=%s hmm=%s points=%s top1=%s seqF1=%s total=%sms score=%sms client=%s",
                trip_id,
                model or "default",
                "on" if use_hmm else "off",
                prediction.get("n_points", "?"),
                _fmt(metrics.get("top1_accuracy")),
                _fmt(metrics.get("sequence_f1")),
                _fmt_ms(total_ms),
                _fmt_ms(score_ms),
                client or "-",
            )

        # The contract's shape: every predict_trip key plus a server-side "metrics" object.
        return {**prediction, "metrics": metrics}

    # ------------------------------------------------------------------ passthrough

    async def models(self) -> list[dict]:
        return await run_in_threadpool(list_models)

    async def experiments(self) -> dict:
        return await run_in_threadpool(experiment_history)


ml_adapter = MLAdapter()
