"""Step 7: HMM + Viterbi route reconstruction.

Hidden states   the candidate arcs of each GPS observation (cands rows, so every state maps
                back to exactly one arc id).
Emission        w_e * log P(candidate | point), with the classifier's pair probabilities
                normalised over the point's candidates.
Transition      Newson & Krumm (2009): log p = -|d_route - d_gps| / beta, where d_gps is the
                straight-line distance between the two GPS points and d_route the directed
                network distance between the two candidates' projected positions:
                    same arc, moving forward     (t_b - t_a) * L
                    otherwise                    (1 - t_a) * L_a + SP(end of a, start of b) + t_b * L_b
                SP = Dijkstra on the corridor graph, cached per source node, cut at sp_limit.
                Unreachable pairs get a fixed floor (unreachable_logp) instead of -inf, so the
                decoder can still pass a hole in the network rather than fail.
Gaps            the chain is split where dt > gap_dt or the GPS jump > gap_dist; each piece is
                decoded on its own (a 42-hour gap says nothing about which road comes next).
Points without candidates are skipped by the lattice and reported as unmatched (-1).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .features import TripContext


@dataclass
class HMMParams:
    beta: float = 10.0
    emission_weight: float = 1.0
    unreachable_logp: float = -40.0
    sp_limit: float = 1500.0
    gap_dt: float = 60.0
    gap_dist: float = 1000.0
    backward_tol: float = 15.0

    def as_dict(self):
        return asdict(self)


class RoadGraph:
    """Directed corridor graph with cached single-source shortest paths."""

    def __init__(self, ctx: TripContext, sp_limit: float):
        net = ctx.net
        nodes, inv = np.unique(np.r_[net.u, net.v], return_inverse=True)
        k = len(net.u)
        self.lu, self.lv = inv[:k], inv[k:]
        w = np.maximum(net.length, 0.01)
        if k:
            order = np.lexsort((w, self.lv, self.lu))
            su, sv = self.lu[order], self.lv[order]
            first = np.r_[True, (su[1:] != su[:-1]) | (sv[1:] != sv[:-1])]
            keep = order[first]  # shortest arc for every (u, v) pair
        else:  # empty corridor: no road anywhere near the track
            keep = np.zeros(0, dtype=np.int64)
        self.n = len(nodes)
        self.G = csr_matrix((w[keep], (self.lu[keep], self.lv[keep])), shape=(self.n, self.n))
        self.edge_arc = {(int(a), int(b)): int(i) for a, b, i in zip(self.lu[keep], self.lv[keep], keep)}
        self.sp_limit = sp_limit
        self._dist: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._pred: dict[int, np.ndarray] = {}

    def dist(self, src: int, targets: np.ndarray) -> np.ndarray:
        if src not in self._dist:
            d = dijkstra(self.G, directed=True, indices=src, limit=self.sp_limit)
            idx = np.flatnonzero(np.isfinite(d))
            self._dist[src] = (idx, d[idx])
        idx, vals = self._dist[src]
        pos = np.searchsorted(idx, targets)
        pos_c = np.minimum(pos, len(idx) - 1)
        hit = (pos < len(idx)) & (idx[pos_c] == targets)
        return np.where(hit, vals[pos_c], np.inf)

    def path_arcs(self, src: int, dst: int) -> list[int] | None:
        """Corridor-local arc indices of the shortest path src -> dst (None if unreachable)."""
        if src == dst:
            return []
        if src not in self._pred:
            _, pred = dijkstra(self.G, directed=True, indices=src, limit=self.sp_limit,
                               return_predecessors=True)
            self._pred[src] = pred
        pred = self._pred[src]
        if pred[dst] < 0:
            return None
        nodes = [dst]
        while nodes[-1] != src:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        return [self.edge_arc[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]


def route_distance(graph: RoadGraph, ctx: TripContext, la, ta, lb, tb, tol: float) -> np.ndarray:
    """Directed network distance (m) from positions (la, ta) to (lb, tb); arrays broadcast as
    (n_a, 1) x (1, n_b). inf where unreachable within the search limit."""
    L = ctx.net.length
    la = np.asarray(la)[:, None]
    ta = np.asarray(ta)[:, None]
    lb = np.asarray(lb)[None, :]
    tb = np.asarray(tb)[None, :]
    va = graph.lv[la[:, 0]]
    ub = graph.lu[lb[0]]
    sp = np.full((la.shape[0], lb.shape[1]), np.inf)
    for s in np.unique(va):
        sp[va == s] = graph.dist(int(s), ub)[None, :]
    d_gen = (1.0 - ta) * L[la] + sp + tb * L[lb]
    along = (tb - ta) * L[la]
    same_fwd = (la == lb) & (along >= -tol)
    return np.where(same_fwd, np.abs(along), d_gen)


def _norm_log(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, None)
    return np.log(p / p.sum())


def viterbi(ctx: TripContext, prob: np.ndarray, params: HMMParams, graph: RoadGraph | None = None):
    """Best candidate row per observation. Returns (row index per point or -1, graph)."""
    c = ctx.cands
    graph = graph or RoadGraph(ctx, params.sp_limit)
    n = len(ctx.px)
    best_row = np.full(n, -1, dtype=np.int64)
    if len(c.obs) == 0:
        return best_row, graph
    starts = np.flatnonzero(np.r_[True, c.obs[1:] != c.obs[:-1]])
    ends = np.r_[starts[1:], len(c.obs)]
    obs_list = c.obs[starts]

    def finish(chain, final_score):
        # chain: list of (obs, rows, backpointer); backtrack from the best final state
        j = int(np.argmax(final_score))
        for k in range(len(chain) - 1, -1, -1):
            o, rows, bp = chain[k]
            best_row[o] = rows[j]
            if bp is not None:
                j = int(bp[j])

    seg = segment_ids(ctx, params)
    chain = []
    score = None
    prev = None
    for o, s, e in zip(obs_list, starts, ends):
        rows = np.arange(s, e)
        emis = params.emission_weight * _norm_log(prob[s:e])
        if prev is not None:
            po, prows = prev
            d_gps = float(np.hypot(ctx.px[o] - ctx.px[po], ctx.py[o] - ctx.py[po]))
            if seg[o] != seg[po]:  # GPS gap: decode the next piece independently
                finish(chain, score)
                chain, score = [], None
        if score is None:
            score = emis
            chain.append((o, rows, None))
        else:
            po, prows = prev
            d_route = route_distance(graph, ctx, c.local[prows], c.t[prows], c.local[rows], c.t[rows],
                                     params.backward_tol)
            logt = np.where(np.isfinite(d_route), -np.abs(d_route - d_gps) / params.beta,
                            params.unreachable_logp)
            tot = score[:, None] + logt
            bp = np.argmax(tot, axis=0)
            score = tot[bp, np.arange(len(rows))] + emis
            chain.append((o, rows, bp))
        prev = (o, rows)
    finish(chain, score)
    return best_row, graph


def segment_ids(ctx: TripContext, params: HMMParams) -> np.ndarray:
    """Evidence segment of every point (-1 = no candidates). A new segment starts where two
    consecutive points that have candidates are separated by dt > gap_dt or a jump > gap_dist.
    The decoder, route assembly and continuity all use this, so nothing is joined across a gap."""
    seg = np.full(len(ctx.px), -1, dtype=np.int64)
    obs = np.unique(ctx.cands.obs)
    if len(obs) == 0:
        return seg
    dt = np.diff(ctx.t[obs])
    jump = np.hypot(np.diff(ctx.px[obs]), np.diff(ctx.py[obs]))
    seg[obs] = np.cumsum(np.r_[False, (dt > params.gap_dt) | (jump > params.gap_dist)])
    return seg


def plausible_continuity(ctx: TripContext, graph: RoadGraph, rows_per_point: np.ndarray,
                         seg: np.ndarray | None = None, factor: float = 3.0,
                         slack: float = 100.0) -> tuple[float | None, int]:
    """(share, number of transitions checked). A transition is a change of predicted arc
    between consecutive matched points of the same evidence segment; it counts as continuous
    if the next arc is reachable over the network within max(factor * GPS distance, slack).
    With no transition to check the share is None (not 1.0): "nothing to judge" is not
    "perfectly continuous"."""
    c = ctx.cands
    pts = np.flatnonzero(rows_per_point >= 0)
    if len(pts) < 2:
        return None, 0
    r = rows_per_point[pts]
    same_seg = np.ones(len(pts) - 1, dtype=bool) if seg is None else (seg[pts[1:]] == seg[pts[:-1]])
    change = np.flatnonzero((c.local[r[1:]] != c.local[r[:-1]]) & same_seg)
    if len(change) == 0:
        return None, 0
    ok = 0
    for k in change:
        ra, rb = r[k], r[k + 1]
        pa, pb = pts[k], pts[k + 1]
        d_gps = float(np.hypot(ctx.px[pb] - ctx.px[pa], ctx.py[pb] - ctx.py[pa]))
        d = route_distance(graph, ctx, [c.local[ra]], [c.t[ra]], [c.local[rb]], [c.t[rb]], 15.0)[0, 0]
        ok += bool(d <= max(factor * d_gps, slack))
    return ok / len(change), int(len(change))


def build_route(ctx: TripContext, graph: RoadGraph, rows_per_point: np.ndarray,
                seg: np.ndarray | None = None) -> list[list[int]]:
    """Route pieces as lists of ORIGINAL arc ids, in travel order. Within an evidence segment,
    consecutive distinct predicted arcs are joined by the shortest network path; a new piece
    starts at every GPS gap (no line is drawn where there is no evidence) and wherever the
    join is unreachable."""
    c = ctx.cands
    pts = np.flatnonzero(rows_per_point >= 0)
    if len(pts) == 0:
        return []
    arcs = c.local[rows_per_point[pts]]
    segs = seg[pts] if seg is not None else np.zeros(len(pts), dtype=np.int64)
    keep = np.r_[True, (arcs[1:] != arcs[:-1]) | (segs[1:] != segs[:-1])]
    arcs, segs = arcs[keep], segs[keep]
    pieces, cur = [], [int(arcs[0])]
    for i in range(1, len(arcs)):
        a, b = int(arcs[i - 1]), int(arcs[i])
        path = None if segs[i] != segs[i - 1] else graph.path_arcs(int(graph.lv[a]), int(graph.lu[b]))
        if path is None:
            pieces.append(cur)
            cur = [b]
        else:
            cur.extend(path)
            cur.append(b)
    pieces.append(cur)
    ids = ctx.net.arc_ids
    return [[int(ids[i]) for i in p] for p in pieces]
