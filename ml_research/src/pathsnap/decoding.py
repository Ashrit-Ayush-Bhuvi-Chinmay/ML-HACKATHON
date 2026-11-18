"""HMM-style route reconstruction with Viterbi decoding."""

from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence

import numpy as np


def viterbi(emissions: Sequence[Mapping[Hashable, float]],
            transition_log_prob: Mapping[tuple[Hashable, Hashable], float],
            default_transition_log_prob: float = -20.0) -> list[Hashable]:
    """Decode the highest-scoring state sequence from per-time emission probabilities."""
    if not emissions:
        return []
    if any(not row for row in emissions):
        raise ValueError("Every observation must have at least one candidate state.")
    scores: dict[Hashable, float] = {s: math.log(max(float(p), 1e-300)) for s, p in emissions[0].items()}
    backpointers: list[dict[Hashable, Hashable]] = []
    for row in emissions[1:]:
        current: dict[Hashable, float] = {}
        back: dict[Hashable, Hashable] = {}
        for state, probability in row.items():
            best_prev, best_score = None, -math.inf
            for prev, prev_score in scores.items():
                score = prev_score + transition_log_prob.get((prev, state), default_transition_log_prob)
                if score > best_score:
                    best_prev, best_score = prev, score
            if best_prev is not None:
                current[state] = best_score + math.log(max(float(probability), 1e-300))
                back[state] = best_prev
        scores = current
        backpointers.append(back)
    if not scores:
        raise ValueError("No reachable candidate sequence.")
    state = max(scores, key=scores.get)
    path = [state]
    for back in reversed(backpointers):
        state = back[state]
        path.append(state)
    return list(reversed(path))


def adjacency_log_probabilities(edges: Sequence[Mapping], stay_log_prob: float = -1.0,
                                connected_log_prob: float = -2.0,
                                disconnected_log_prob: float = -12.0) -> dict[tuple[str, str], float]:
    """Return sparse transition scores using directed edge connectivity.

    Missing transitions use the caller's disconnected default. This avoids constructing
    a quadratic matrix for real road networks.
    """
    by_id = {str(e["edge_id"]): e for e in edges}
    outgoing: dict[str, list[str]] = {}
    for edge_id, edge in by_id.items():
        outgoing.setdefault(str(edge["u"]), []).append(edge_id)
    scores: dict[tuple[str, str], float] = {(edge_id, edge_id): stay_log_prob for edge_id in by_id}
    for edge_id, edge in by_id.items():
        for next_id in outgoing.get(str(edge["v"]), ()):
            scores[(edge_id, next_id)] = connected_log_prob
    return scores


def decode_candidates(candidate_rows: Sequence[Sequence[Mapping]],
                      edge_probabilities: Sequence[Sequence[float]],
                      transition_log_prob: Mapping[tuple[Hashable, Hashable], float]) -> list[Hashable]:
    if len(candidate_rows) != len(edge_probabilities):
        raise ValueError("Candidate and probability sequences must align.")
    emissions = []
    for candidates, probabilities in zip(candidate_rows, edge_probabilities):
        if len(candidates) != len(probabilities):
            raise ValueError("Each point's candidates and probabilities must align.")
        emissions.append({str(c["edge_id"]): float(p) for c, p in zip(candidates, probabilities)})
    return viterbi(emissions, transition_log_prob)


def _decode_trip_segment(data, probabilities: np.ndarray, point_indices: list[int],
                         same_arc_score: float, connected_score: float,
                         disconnected_score: float, distance_ratio_weight: float) -> list[int]:
    first = point_indices[0]
    start, stop = int(data.point_offsets[first]), int(data.point_offsets[first + 1])
    first_edges = data.candidate_edge_ids[start:stop]
    first_emissions = probabilities[start:stop].astype(np.float64)
    first_emissions = first_emissions / max(float(first_emissions.sum()), 1e-300)
    scores = np.log(np.clip(first_emissions, 1e-300, 1.0))
    state_edges = first_edges
    state_edges_by_time = [state_edges]
    backpointers: list[np.ndarray] = []
    for point_idx in point_indices[1:]:
        cur_start, cur_stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx + 1])
        current_edges = data.candidate_edge_ids[cur_start:cur_stop]
        current_emissions = probabilities[cur_start:cur_stop].astype(np.float64)
        current_emissions = current_emissions / max(float(current_emissions.sum()), 1e-300)
        current_log_emissions = np.log(np.clip(current_emissions, 1e-300, 1.0))

        previous_by_to: dict[int, list[int]] = {}
        previous_by_edge: dict[int, int] = {}
        for prev_idx, edge_id in enumerate(state_edges):
            edge_id = int(edge_id)
            previous_by_edge[edge_id] = prev_idx
            previous_by_to.setdefault(int(data.arcs[edge_id, 1]), []).append(prev_idx)
        score_order = np.argsort(scores)[::-1]
        gps_step = max(float(data.point_observed_steps_m[point_idx]), 0.0)
        pointers = np.full(len(current_edges), -1, dtype=np.int32)
        next_scores = np.full(len(current_edges), -np.inf, dtype=np.float64)

        for cur_idx, raw_edge_id in enumerate(current_edges):
            edge_id = int(raw_edge_id)
            from_node = int(data.arcs[edge_id, 0])
            same_idx = previous_by_edge.get(edge_id)
            connected_previous = previous_by_to.get(from_node, [])
            eligible: set[int] = set(connected_previous)
            if same_idx is not None:
                eligible.add(same_idx)
            best_score, best_prev = -np.inf, -1

            transition_indices = set(connected_previous)
            if same_idx is not None:
                transition_indices.add(same_idx)
            for prev_idx in transition_indices:
                prev_edge = int(state_edges[prev_idx])
                if prev_edge == edge_id:
                    base_score = same_arc_score
                    network_distance = abs(float(data.candidate_along_fractions[cur_start + cur_idx]) -
                                            float(data.candidate_along_fractions[
                                                int(data.point_offsets[point_idx - 1]) + prev_idx])) * \
                                      float(data.candidate_arc_lengths_m[cur_start + cur_idx])
                else:
                    base_score = connected_score
                    previous_row = int(data.point_offsets[point_idx - 1]) + prev_idx
                    current_row = cur_start + cur_idx
                    network_distance = (float(data.candidate_arc_lengths_m[previous_row]) *
                                        (1.0 - float(data.candidate_along_fractions[previous_row])) +
                                        float(data.candidate_arc_lengths_m[current_row]) *
                                        float(data.candidate_along_fractions[current_row]))
                ratio_penalty = distance_ratio_weight * abs(math.log(
                    (network_distance + 5.0) / (gps_step + 5.0)))
                candidate_score = scores[prev_idx] + base_score - ratio_penalty
                if candidate_score > best_score:
                    best_score, best_prev = candidate_score, prev_idx

            # Disconnected moves remain possible, but incur a finite, strong penalty.
            for prev_idx in score_order:
                prev_idx = int(prev_idx)
                if prev_idx not in eligible:
                    candidate_score = scores[prev_idx] + disconnected_score
                    if candidate_score > best_score:
                        best_score, best_prev = candidate_score, prev_idx
                    break
            if best_prev >= 0:
                pointers[cur_idx] = best_prev
                next_scores[cur_idx] = best_score + current_log_emissions[cur_idx]
        if np.all(pointers < 0):
            # Defensive reset for malformed edge tables or numerical failures.
            return [int(state_edges[int(np.argmax(scores))])] + [
                int(current_edges[int(np.argmax(current_log_emissions))]) for _ in point_indices[1:]]
        backpointers.append(pointers)
        scores, state_edges = next_scores, current_edges
        state_edges_by_time.append(state_edges)

    state_idx = int(np.argmax(scores))
    state_indices = [state_idx]
    for pointers in reversed(backpointers):
        state_idx = int(pointers[state_idx])
        state_indices.append(state_idx)
    state_indices.reverse()
    return [int(state_edges_by_time[t][idx]) for t, idx in enumerate(state_indices)]


def decode_trip_hmm(data, candidate_probabilities: Sequence[float],
                    same_arc_score: float = -0.2, connected_score: float = -0.7,
                    disconnected_score: float = -12.0,
                    distance_ratio_weight: float = 0.6) -> np.ndarray:
    """Decode a feature-cached trip; empty points and long gaps reset the chain.

    Candidate probabilities are normalized within each observation and used in log
    space. Network transitions reward staying on an arc or moving to a directed
    successor; disconnected moves remain possible with a strong finite penalty.
    """
    probabilities = np.asarray(candidate_probabilities, dtype=np.float64)
    if probabilities.shape != (data.candidate_count,):
        raise ValueError("Candidate probabilities must align with the trip's candidate rows")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("Candidate probabilities must be finite and nonnegative")
    decoded = np.full(data.point_count, -1, dtype=np.int64)
    segments: list[list[int]] = []
    current: list[int] = []
    for point_idx in range(data.point_count):
        if data.point_gap_resets[point_idx] and current:
            segments.append(current)
            current = []
        start, stop = int(data.point_offsets[point_idx]), int(data.point_offsets[point_idx + 1])
        if start == stop:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(point_idx)
    if current:
        segments.append(current)
    for segment in segments:
        path = _decode_trip_segment(data, probabilities, segment, same_arc_score,
                                    connected_score, disconnected_score, distance_ratio_weight)
        decoded[segment] = path
    return decoded
