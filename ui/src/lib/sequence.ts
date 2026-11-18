/**
 * Descriptive statistics of the predicted arc sequence.
 *
 * These describe the shape of the prediction itself (how many points were matched,
 * how many arc changes). They are NOT evaluation metrics: accuracy, precision,
 * recall, F1 and continuity are computed by the API against the reference route,
 * which the UI never sees.
 */

export interface ArcRun {
  /** Arc id, or null for an unmatched stretch. */
  arc: number | null
  /** First GPS point index (inclusive). */
  start: number
  /** Last GPS point index (inclusive). */
  end: number
  length: number
}

export interface SequenceStats {
  total: number
  matched: number
  unmatched: number
  distinctArcs: number
  matchedRuns: number
  unmatchedRuns: number
  longestUnmatched: number
}

export function runsOf(arcs: (number | null)[]): ArcRun[] {
  const runs: ArcRun[] = []
  for (let i = 0; i < arcs.length; i++) {
    const a = arcs[i]
    const last = runs[runs.length - 1]
    if (last && last.arc === a) {
      last.end = i
      last.length++
    } else {
      runs.push({ arc: a, start: i, end: i, length: 1 })
    }
  }
  return runs
}

export function statsOf(arcs: (number | null)[], runs: ArcRun[]): SequenceStats {
  let matched = 0
  const distinct = new Set<number>()
  for (const a of arcs) {
    if (a !== null) {
      matched++
      distinct.add(a)
    }
  }
  let matchedRuns = 0
  let unmatchedRuns = 0
  let longestUnmatched = 0
  for (const r of runs) {
    if (r.arc === null) {
      unmatchedRuns++
      longestUnmatched = Math.max(longestUnmatched, r.length)
    } else matchedRuns++
  }
  return {
    total: arcs.length,
    matched,
    unmatched: arcs.length - matched,
    distinctArcs: distinct.size,
    matchedRuns,
    unmatchedRuns,
    longestUnmatched,
  }
}

/** The run containing GPS point `i` (binary search over runs ordered by start). */
export function runAt(runs: ArcRun[], i: number): ArcRun | null {
  let lo = 0
  let hi = runs.length - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    const r = runs[mid]
    if (i < r.start) hi = mid - 1
    else if (i > r.end) lo = mid + 1
    else return r
  }
  return null
}

/** Index of the next (dir 1) or previous (dir -1) unmatched run start relative to point `from`. */
export function adjacentGap(runs: ArcRun[], from: number, dir: 1 | -1): number | null {
  if (dir === 1) {
    for (const r of runs) if (r.arc === null && r.start > from) return r.start
  } else {
    for (let k = runs.length - 1; k >= 0; k--) {
      const r = runs[k]
      if (r.arc === null && r.end < from) return r.start
    }
  }
  return null
}

/** Positions (0-based) at which each arc id appears in route_arcs. */
export function routePositions(routeArcs: number[]): Map<number, number[]> {
  const m = new Map<number, number[]>()
  routeArcs.forEach((a, i) => {
    const list = m.get(a)
    if (list) list.push(i)
    else m.set(a, [i])
  })
  return m
}
