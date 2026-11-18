"""Final evaluator-only reporting; labels are requested only at scoring time."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def score_final_evaluator(y_true, classifier_predictions, decoded_predictions,
                          output_path: str | Path) -> dict[str, Any]:
    """Score frozen predictions against evaluator labels at the final, one-way stage."""
    def summary(pred):
        return {"accuracy": float(accuracy_score(y_true, pred)),
                "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
                "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0))}
    result = {"partition": "evaluator_final_only", "classifier_only": summary(classifier_predictions),
              "hmm_viterbi_decoded": summary(decoded_predictions)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
