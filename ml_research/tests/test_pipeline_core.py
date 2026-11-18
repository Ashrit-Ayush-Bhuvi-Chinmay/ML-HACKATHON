import unittest

from pathsnap.decoding import decode_trip_hmm, viterbi
from pathsnap.dataset import TripData
from pathsnap.feature_pipeline import FEATURE_COLUMNS, TripFeatureSet, build_trip_features
from pathsnap.features import validate_feature_columns
from pathsnap.geo import project_utm
from pathsnap.profiling import profile_dataset
from pathsnap.splitting import EvaluatorPartition, make_group_split, prepare_partitions


class PipelineCoreTests(unittest.TestCase):
  def test_profile_handles_absent_dataset(self):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
      report = profile_dataset(Path(folder) / "missing")
    self.assertFalse(report["exists"])


  def test_split_is_group_disjoint_and_reproducible(self):
    groups = [f"trip-{i}" for i in range(30)]
    a = make_group_split(groups, seed=12)
    b = make_group_split(reversed(groups), seed=12)
    self.assertEqual(a, b)
    self.assertFalse(set(a.train_groups) & set(a.evaluator_groups))
    self.assertTrue(set(a.evaluator_groups))


  def test_evaluator_labels_require_final_evaluation(self):
    partition = EvaluatorPartition(inputs=[1], group_ids=("held-out",), _labels_for_final_scoring=["edge-a"])
    with self.assertRaises(PermissionError):
      partition.final_labels()
    self.assertEqual(partition.final_labels(final_evaluation=True), ["edge-a"])

  def test_prepare_partitions_hides_evaluator_labels_from_inputs(self):
    manifest = make_group_split(["t1", "t2", "t3"], train_fraction=1/3, development_fraction=1/3)
    observations = [{"trajectory_id": group, "point_index": 0, "latitude": 1, "longitude": 2,
                    "edge_id": "leak", "is_match": 1} for group in
                   manifest.train_groups + manifest.development_groups + manifest.evaluator_groups]
    labels = [{"trajectory_id": group, "point_index": 0, "edge_id": f"truth-{group}"} for group in
              manifest.train_groups + manifest.development_groups + manifest.evaluator_groups]
    bundle = prepare_partitions(observations, labels, manifest)
    self.assertNotIn("edge_id", bundle.evaluator.inputs[0])
    self.assertNotIn("is_match", bundle.evaluator.inputs[0])
    self.assertNotIn("truth-" + manifest.evaluator_groups[0], bundle.evaluator.inputs[0].values())
    with self.assertRaises(PermissionError):
      bundle.evaluator.final_labels()
    self.assertTrue(bundle.evaluator.final_labels(final_evaluation=True))


  def test_ground_truth_columns_cannot_be_features(self):
    with self.assertRaises(ValueError):
      validate_feature_columns(["distance_m", "is_match"])


  def test_viterbi_respects_transition_scores(self):
    emissions = [{"a": .95, "b": .05}, {"a": .1, "b": .9}]
    transitions = {("a", "a"): -1, ("a", "b"): -8, ("b", "a"): -8, ("b", "b"): -1}
    self.assertEqual(viterbi(emissions, transitions), ["a", "a"])

  def test_hmm_rewards_connected_candidates_and_resets_on_gaps_or_empty_sets(self):
    import numpy as np
    data = TripFeatureSet(
      split="train", trip_id="hmm-toy", features=np.zeros((4, len(FEATURE_COLUMNS)), np.float32),
      labels=np.asarray([1, 0, 1, 0], np.int8), candidate_edge_ids=np.asarray([0, 1, 0, 1]),
      candidate_from_nodes=np.asarray([0, 2, 0, 2]), candidate_to_nodes=np.asarray([1, 3, 1, 3]),
      candidate_point_indices=np.asarray([0, 0, 1, 1]), candidate_distances_m=np.asarray([1, 2, 1, 2]),
      candidate_along_fractions=np.asarray([.1, .1, .2, .2]),
      candidate_arc_lengths_m=np.asarray([100, 100, 100, 100]),
      point_offsets=np.asarray([0, 2, 4]), point_timestamps_s=np.asarray([1., 2.]),
      point_true_edges=np.asarray([0, 0]), point_route_positions=np.asarray([0, 1]),
      point_route_distances_m=np.asarray([1., 1.]), point_observed_steps_m=np.asarray([0., 10.]),
      point_gap_resets=np.asarray([False, False]), point_candidate_misses=np.asarray([False, False]),
      arcs=np.asarray([[0, 1], [2, 3]]), epsg="EPSG:32632")
    probabilities = np.asarray([.9, .1, .1, .9])
    decoded = decode_trip_hmm(data, probabilities, disconnected_score=-12.0)
    self.assertEqual(decoded.tolist(), [0, 0])
    data.point_gap_resets[1] = True
    self.assertEqual(decode_trip_hmm(data, probabilities).tolist(), [0, 1])
    data.point_offsets = np.asarray([0, 2, 2])
    data.candidate_edge_ids = data.candidate_edge_ids[:2]
    data.candidate_along_fractions = data.candidate_along_fractions[:2]
    data.candidate_arc_lengths_m = data.candidate_arc_lengths_m[:2]
    self.assertEqual(decode_trip_hmm(data, probabilities[:2]).tolist(), [0, -1])


  def test_utm_projection_uses_metric_coordinates(self):
    east, north, zone, epsg = project_utm(9.0, 50.0, zone=32)
    self.assertEqual(zone, 32)
    self.assertEqual(epsg, "EPSG:32632")
    self.assertAlmostEqual(east, 500000.0, delta=1.0)
    self.assertGreater(north, 5_000_000)

  def test_candidate_generation_uses_utm_and_keeps_labels_out_of_features(self):
    import numpy as np
    trip = TripData(
      "train", "toy",
      np.asarray([[9.0001, 50.0, 1.0], [9.0002, 50.0, 2.0]]),
      np.asarray([[9.0, 50.0], [9.001, 50.0],
                  [9.0, 50.0005], [9.001, 50.0005]]),
      np.asarray([[0, 1], [2, 3]], dtype=np.int64),
      np.asarray([0], dtype=np.int64),
    )
    result = build_trip_features(trip, radius_m=100, max_candidates=None)
    self.assertEqual(result.epsg, "EPSG:32632")
    self.assertEqual(result.point_count, 2)
    self.assertEqual(result.candidate_count, 4)
    self.assertEqual(np.count_nonzero(result.labels == 1), 2)
    self.assertFalse(np.any(result.point_candidate_misses))
    self.assertEqual(result.features.shape[1], len(FEATURE_COLUMNS))
    self.assertNotIn("is_match", FEATURE_COLUMNS)
    nearest_gap = FEATURE_COLUMNS.index("nearest_candidate_gap_m")
    previous_delta = FEATURE_COLUMNS.index("previous_projection_delta_m")
    previous_connection = FEATURE_COLUMNS.index("connects_previous_candidate")
    self.assertEqual(float(result.features[0, nearest_gap]), 0.0)
    self.assertGreater(float(result.features[1, nearest_gap]), 10.0)
    self.assertGreater(float(result.features[2, previous_delta]), 0.0)
    self.assertEqual(float(result.features[2, previous_connection]), 1.0)
