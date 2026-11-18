/** PathSnap mark: scattered GPS fixes snapping onto a road line. */
export function BrandMark({ className = 'size-7' }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#1b2a33" />
      <path d="M5 23 C 11 22, 13 12, 19 11 S 26 9, 27 8" fill="none" stroke="#34d6e0" strokeWidth="2.6" strokeLinecap="round" />
      <circle cx="8.5" cy="17.5" r="1.7" fill="#4c8dff" />
      <circle cx="14" cy="20" r="1.7" fill="#4c8dff" />
      <circle cx="17.5" cy="7.5" r="1.7" fill="#4c8dff" />
      <circle cx="24" cy="13.5" r="1.7" fill="#4c8dff" />
    </svg>
  )
}
