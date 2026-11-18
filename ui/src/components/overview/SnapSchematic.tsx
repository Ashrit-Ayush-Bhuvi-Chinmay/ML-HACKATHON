/**
 * Illustration of the map-matching idea: noisy GPS fixes around a street grid, and the route
 * snapped onto the roads. Deterministic geometry, clearly captioned as a schematic: it is not
 * trip data and nothing in it comes from the API.
 */
const ROUTE: [number, number][] = [
  [36, 292],
  [168, 292],
  [168, 188],
  [332, 188],
  [332, 92],
  [524, 92],
]

const STREETS = [
  'M20 292 H540',
  'M20 188 H540',
  'M20 92 H540',
  'M20 240 H300',
  'M260 140 H540',
  'M96 40 V330',
  'M168 40 V330',
  'M250 40 V330',
  'M332 40 V330',
  'M430 40 V330',
  'M500 150 V330',
]

/** Small deterministic pseudo-random generator so the drawing is identical on every render. */
function lcg(seed: number) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) % 4294967296
    return s / 4294967296
  }
}

function sampleFixes() {
  const rnd = lcg(7)
  const pts: { x: number; y: number; off: boolean }[] = []
  for (let i = 0; i < ROUTE.length - 1; i++) {
    const [x1, y1] = ROUTE[i]
    const [x2, y2] = ROUTE[i + 1]
    const len = Math.hypot(x2 - x1, y2 - y1)
    const steps = Math.max(2, Math.round(len / 17))
    for (let k = 0; k < steps; k++) {
      const t = k / steps
      const jitter = 10
      pts.push({ x: x1 + (x2 - x1) * t + (rnd() - 0.5) * jitter * 2, y: y1 + (y2 - y1) * t + (rnd() - 0.5) * jitter * 2, off: false })
    }
  }
  // A few fixes that jumped across buildings: they stay unmatched.
  pts.push({ x: 214, y: 240, off: true }, { x: 226, y: 252, off: true }, { x: 402, y: 140, off: true })
  return pts
}

const FIXES = sampleFixes()
const ROUTE_D = 'M' + ROUTE.map(([x, y]) => `${x} ${y}`).join(' L')

export function SnapSchematic() {
  return (
    <figure className="relative overflow-hidden rounded-[12px] border border-hairline bg-inset">
      <svg viewBox="0 0 560 360" className="block h-auto w-full" role="img" aria-labelledby="snap-title">
        <title id="snap-title">Schematic: noisy GPS fixes around a street grid, with the matched route drawn along the roads</title>
        <defs>
          <pattern id="snap-blocks" width="8" height="8" patternUnits="userSpaceOnUse">
            <rect width="8" height="8" fill="var(--ps-inset)" />
            <circle cx="1" cy="1" r="0.7" fill="var(--ps-hairline)" />
          </pattern>
        </defs>
        <rect width="560" height="360" fill="url(#snap-blocks)" />
        <g stroke="var(--ps-raised)" strokeWidth="9" strokeLinecap="round" fill="none">
          {STREETS.map((d) => (
            <path key={d} d={d} />
          ))}
        </g>
        <g stroke="var(--ps-road)" strokeWidth="1.2" strokeLinecap="round" fill="none" opacity="0.8">
          {STREETS.map((d) => (
            <path key={d} d={d} />
          ))}
        </g>

        <g className="ps-snap-fixes">
          {FIXES.map((p, i) =>
            p.off ? (
              <circle key={i} cx={p.x} cy={p.y} r="4" fill="none" stroke="var(--ps-unmatched)" strokeWidth="1.4" opacity="0.9" />
            ) : (
              <circle key={i} cx={p.x} cy={p.y} r="3.3" fill="var(--ps-gps)" opacity="0.95" />
            ),
          )}
        </g>

        <path d={ROUTE_D} fill="none" stroke="var(--ps-route-casing, #0e151d)" strokeWidth="9" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" className="ps-snap-route" pathLength={1} />
        <path d={ROUTE_D} fill="none" stroke="var(--ps-route)" strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" className="ps-snap-route" pathLength={1} />

        <g fontFamily="var(--font-sans)" fontSize="12.5" className="ps-snap-labels">
          <line x1="226" y1="252" x2="226" y2="318" stroke="var(--ps-ink-faint)" strokeWidth="1" />
          <text x="232" y="322" fill="var(--ps-ink-muted)">fix that jumped off the road: unmatched</text>
          <line x1="84" y1="282" x2="84" y2="248" stroke="var(--ps-ink-faint)" strokeWidth="1" />
          <text x="40" y="242" fill="var(--ps-ink-muted)">noisy GPS fixes</text>
          <line x1="430" y1="92" x2="430" y2="58" stroke="var(--ps-ink-faint)" strokeWidth="1" />
          <text x="384" y="52" fill="var(--ps-ink)">route on the road network</text>
        </g>
      </svg>
      <figcaption className="absolute right-3 bottom-2.5 rounded bg-surface/90 px-2 py-0.5 text-[11.5px] text-ink-faint">
        Schematic illustration, not trip data
      </figcaption>
    </figure>
  )
}
