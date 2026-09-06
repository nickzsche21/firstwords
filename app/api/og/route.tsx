import { ImageResponse } from "next/og";
import { stageFor } from "@/lib/stages";

export const alt = "FIRST WORDS";
export const size = { width: 1200, height: 630 };

/**
 * Satori quirks that matter here: React Fragments are not laid out, any box
 * with more than one child needs an explicit display, and wrapping text needs a
 * real width rather than just maxWidth.
 */
export function GET(req: Request) {
  const url = new URL(req.url);
  const quote = (url.searchParams.get("w") ?? "").trim().slice(0, 140);
  const loss = Number(url.searchParams.get("l"));
  const stage = Number.isFinite(loss) ? stageFor(loss) : null;

  const body = quote || "Watch a neural network be born in your browser.";
  // Long babble needs to shrink or it overflows the card.
  const fontSize = body.length > 110 ? 40 : body.length > 70 ? 48 : 58;

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          background: "#08090b",
          color: "#e8eaed",
          fontFamily: "monospace",
          padding: "0 88px",
        }}
      >
        <div style={{ fontSize: 21, letterSpacing: 9, color: "#7c8595" }}>
          FIRST WORDS
        </div>
        <div
          style={{
            width: 1024,
            marginTop: 34,
            fontSize,
            lineHeight: 1.42,
            color: "#e8eaed",
          }}
        >
          {quote ? `“${body}”` : body}
        </div>
        <div
          style={{
            width: 1024,
            marginTop: 40,
            fontSize: 23,
            color: "#7dd3a0",
            letterSpacing: 3,
          }}
        >
          {stage
            ? `STAGE: ${stage.label}  ·  ${loss.toFixed(2)} nats/char`
            : "TRAINED FROM SCRATCH IN A BROWSER TAB"}
        </div>
      </div>
    ),
    { ...size }
  );
}
