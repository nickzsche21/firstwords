"use client";

import { useMemo } from "react";

/**
 * The learning curve. Autoscaled to the observed range rather than pinned to
 * zero: a char-level model spends its whole life between about 4.5 and 1.2, and
 * a zero-based axis flattens the only part anyone cares about into a hairline.
 */
export default function LossCurve({
  history,
  width = 520,
  height = 120,
}: {
  history: number[];
  width?: number;
  height?: number;
}) {
  const { d, area, lo, hi } = useMemo(() => {
    if (history.length < 2) return { d: "", area: "", lo: 0, hi: 0 };

    // Downsample to at most one point per horizontal pixel; plotting thousands
    // of steps into 520px is pure path bloat.
    const stride = Math.max(1, Math.floor(history.length / width));
    const pts: number[] = [];
    for (let i = 0; i < history.length; i += stride) pts.push(history[i]);
    if (pts[pts.length - 1] !== history[history.length - 1]) {
      pts.push(history[history.length - 1]);
    }

    const lo = Math.min(...pts);
    const hi = Math.max(...pts);
    const span = Math.max(0.15, hi - lo);
    const x = (i: number) => (i / Math.max(1, pts.length - 1)) * width;
    const y = (v: number) => height - ((v - lo) / span) * (height - 8) - 4;

    const d = pts.map((v, i) => `${i ? "L" : "M"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");
    const area = `${d} L ${width} ${height} L 0 ${height} Z`;
    return { d, area, lo, hi };
  }, [history, width, height]);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full"
        style={{ height }}
        data-points={history.length}
        preserveAspectRatio="none"
        aria-label="training loss over time"
      >
        <defs>
          <linearGradient id="lossfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {area && <path d={area} fill="url(#lossfill)" />}
        {d && (
          <path
            d={d}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="1.5"
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
          />
        )}
      </svg>
      {history.length > 1 && (
        <>
          <span className="absolute left-0 top-0 text-[10px]" style={{ color: "var(--muted)" }}>
            {hi.toFixed(2)}
          </span>
          <span className="absolute bottom-0 left-0 text-[10px]" style={{ color: "var(--muted)" }}>
            {lo.toFixed(2)}
          </span>
        </>
      )}
    </div>
  );
}
