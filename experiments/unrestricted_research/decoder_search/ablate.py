"""Compare the selected blend with the same decoder and its learned term removed."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import pandas as pd

from run import FROZEN, SEARCH, Setting, evaluate, load_predictions, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    results = parser.parse_args().results
    summary = pd.read_csv(results / "summary.csv")
    learned_settings = {s.name: s for s in SEARCH if s.learned_weight}
    selection = summary[(summary.cohort == "train_select") & summary.setting.isin(learned_settings)]
    best = learned_settings[selection.nlargest(1, "sequence_f1").iloc[0]["setting"]]
    control = Setting("blend_matched_control", best.model, best.beta, weight=best.weight,
                      adaptive=best.adaptive)
    learned = SimpleNamespace(booster_=lgb.Booster(model_file=str(results / "transition_model.txt")))
    split = pd.read_csv(results / "train_split.csv", dtype={"trip_id": str})
    train = load_predictions("oof")
    test = load_predictions("test")
    settings = [FROZEN, control, best]
    check = evaluate("train", split.loc[split.role == "check", "trip_id"], train,
                     settings, learned, "train_check")
    student = evaluate("student_test", sorted(test[FROZEN.model].groups), test,
                       settings, learned, "student_test")
    per_trip = pd.concat([check, student])
    per_trip.to_csv(results / "ablation_per_trip.csv", index=False)
    scores = summarize(per_trip)
    scores.to_csv(results / "ablation_summary.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
