"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import LossCurve from "./LossCurve";
import { PRESETS } from "@/lib/presets";
import { STAGES, stageFor, stageIndex } from "@/lib/stages";
import type { TinyModel } from "@/lib/tinymodel";

const MIN_CHARS = 400;
/** first run; "keep going" adds another of these */
const DURATION_MS = 45_000;
const BATCH = 64;

type Phase = "setup" | "training" | "done";

type Vitals = {
  step: number;
  progress: number;
  loss: number;
  params: number;
  vocab: number;
  chars: number;
  backend: string;
  stepsPerSec: number;
};

export default function Nursery() {
  const [phase, setPhase] = useState<Phase>("setup");
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const [sample, setSample] = useState("");
  const [vitals, setVitals] = useState<Vitals | null>(null);
  const [firstWords, setFirstWords] = useState("");

  const stopRef = useRef(false);
  const stepRef = useRef(0);
  const modelRef = useRef<TinyModel | null>(null);
  const ctxRef = useRef<{
    model: TinyModel;
    vocabSize: number;
    corpusChars: number;
    backend: string;
  } | null>(null);
  // Ticks arrive faster than React should re-render; buffer and flush on a timer.
  const bufRef = useRef<{ losses: number[]; sample?: string; v?: Vitals }>({ losses: [] });

  useEffect(() => {
    return () => {
      stopRef.current = true;
      modelRef.current?.dispose();
    };
  }, []);

  const begin = useCallback(async () => {
    setError(null);
    const raw = text.trim();
    if (raw.length < MIN_CHARS) {
      setError(
        `Needs at least ${MIN_CHARS} characters to learn anything — you gave it ${raw.length}. Paste more, or pick a starter below.`
      );
      return;
    }

    setPhase("training");
    setHistory([]);
    setSample("");
    setFirstWords("");

    // tfjs is ~1MB. Nobody pays for it until they actually commit to a run.
    const tm = await import("@/lib/tinymodel");
    const backend = await tm.initBackend();

    const corpus = tm.prepareCorpus(raw);
    const vocab = tm.buildVocab(corpus);
    const data = tm.encode(corpus, vocab);

    const model = new tm.TinyModel(vocab, data);
    modelRef.current = model;
    stopRef.current = false;

    await runTraining(model, vocab.size, corpus.length, backend, 0);
  }, [text]);

  /** One training budget. Called again by "keep going" to extend the same model. */
  const runTraining = useCallback(
    async (
      model: TinyModel,
      vocabSize: number,
      corpusChars: number,
      backend: string,
      startStep: number
    ) => {
      setPhase("training");
      const t0 = performance.now();
      let flushed = t0;

      const finalStep = await model.train({
      durationMs: DURATION_MS,
      batchSize: BATCH,
      startStep,
      shouldStop: () => stopRef.current,
      onTick: (t) => {
        const buf = bufRef.current;
        buf.losses.push(t.smoothed);
        if (t.sample) buf.sample = t.sample;
        buf.v = {
          step: t.step,
          progress: t.progress,
          loss: t.smoothed,
          params: model.params,
          vocab: vocabSize,
          chars: corpusChars,
          backend,
          stepsPerSec: (t.step - startStep) / ((performance.now() - t0) / 1000),
        };

        // Repaint at ~15fps regardless of how fast the GPU is chewing.
        const now = performance.now();
        if (now - flushed > 66) {
          flushed = now;

          // Snapshot before clearing. A functional updater runs when React
          // gets round to it, not when it is queued, so `(h) => [...h,
          // ...buf.losses]` would read the array *after* the line below swapped
          // it for an empty one — silently appending nothing on every flush.
          const pending = buf.losses;
          const sampled = buf.sample;
          buf.losses = [];
          buf.sample = undefined;

          setHistory((h) => [...h, ...pending]);
          if (sampled) setSample(sampled);
          if (buf.v) setVitals(buf.v);
        }
      },
    });

      stepRef.current = finalStep;
      ctxRef.current = { model, vocabSize, corpusChars, backend };

      // Best-of a few draws at a low temperature: any single sample is a coin
      // flip, and the whole share hinges on this one string.
      const candidates = [0.55, 0.65, 0.75].map((t) => model.generate(150, t));
      setFirstWords(pickQuotable(candidates));
      setPhase("done");
    },
    []
  );

  const keepGoing = useCallback(() => {
    const c = ctxRef.current;
    if (!c) return;
    stopRef.current = false;
    void runTraining(c.model, c.vocabSize, c.corpusChars, c.backend, stepRef.current);
  }, [runTraining]);

  const stop = useCallback(() => {
    // The training loop checks this between steps and returns, which lets
    // runTraining fall through to its own sampling and phase change.
    stopRef.current = true;
  }, []);

  const reset = useCallback(() => {
    stopRef.current = true;
    modelRef.current?.dispose();
    modelRef.current = null;
    ctxRef.current = null;
    stepRef.current = 0;
    setPhase("setup");
    setHistory([]);
    setSample("");
    setVitals(null);
    setFirstWords("");
  }, []);

  const loss = vitals?.loss ?? Infinity;
  const stage = stageFor(loss);
  const progress = vitals?.progress ?? 0;

  return (
    <main className="mx-auto flex min-h-dvh max-w-2xl flex-col gap-6 px-5 py-10">
      <Header phase={phase} />

      {phase === "setup" && (
        <Setup
          text={text}
          setText={setText}
          error={error}
          onBegin={begin}
        />
      )}

      {phase !== "setup" && (
        <>
          <StageBar stage={stage} loss={loss} />
          <Babble text={phase === "done" ? firstWords : sample} done={phase === "done"} />
          <div
            className="rounded-lg border p-4"
            style={{ borderColor: "var(--line)", background: "var(--panel)" }}
          >
            <div className="mb-2 flex items-baseline justify-between text-[10px] uppercase tracking-widest"
              style={{ color: "var(--muted)" }}>
              <span>cross-entropy / char</span>
              <span className="tabular-nums" style={{ color: "var(--accent)" }}>
                {Number.isFinite(loss) ? loss.toFixed(3) : "—"}
              </span>
            </div>
            <LossCurve history={history} />
          </div>
          {vitals && <VitalsRow v={vitals} progress={progress} phase={phase} />}
        </>
      )}

      {phase === "training" && (
        <button
          onClick={() => void stop()}
          className="self-start rounded border px-4 py-2 text-xs uppercase tracking-widest transition-colors hover:bg-white/5"
          style={{ borderColor: "var(--line)", color: "var(--muted)" }}
        >
          stop here
        </button>
      )}

      {phase === "done" && (
        <Done
          firstWords={firstWords}
          vitals={vitals}
          onReset={reset}
          onKeepGoing={keepGoing}
        />
      )}

      <Footer />
    </main>
  );
}

/* ------------------------------------------------------------------ */

function Header({ phase }: { phase: Phase }) {
  return (
    <header className="space-y-2">
      <h1 className="text-2xl font-bold tracking-tight" style={{ color: "var(--ink)" }}>
        FIRST WORDS
      </h1>
      <p className="sans text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
        {phase === "setup" && (
          <>
            Give it some text and a neural network is born in this tab and raised
            on it. You will watch it go from static, to letters, to words, to
            something that sounds like you. Nothing is uploaded — the model is
            trained on your machine and dies when you close the page.
          </>
        )}
        {phase === "training" && <>Training locally. Do not close the tab.</>}
        {phase === "done" && (
          <>
            This is the best it managed in the time it had. It is still a
            newborn — every extra run makes it measurably less confused.
          </>
        )}
      </p>
    </header>
  );
}

function Setup({
  text,
  setText,
  error,
  onBegin,
}: {
  text: string;
  setText: (s: string) => void;
  error: string | null;
  onBegin: () => void;
}) {
  return (
    <div className="space-y-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={9}
        spellCheck={false}
        placeholder="Paste anything. Your tweets. A group chat export. Your dissertation. The longer and more repetitive your voice, the better the impression."
        className="w-full resize-y rounded-lg border p-4 text-[13px] leading-relaxed outline-none focus:border-[var(--accent)]"
        style={{ borderColor: "var(--line)", background: "var(--panel)", color: "var(--ink)" }}
      />

      <div className="flex items-center justify-between text-[11px]" style={{ color: "var(--muted)" }}>
        <span className="tabular-nums">
          {text.trim().length.toLocaleString()} chars
          {text.trim().length > 0 && text.trim().length < MIN_CHARS && ` · need ${MIN_CHARS}`}
        </span>
        <button
          onClick={onBegin}
          disabled={text.trim().length < MIN_CHARS}
          className="rounded px-5 py-2 text-xs font-bold uppercase tracking-widest text-black transition-opacity disabled:opacity-25"
          style={{ background: "var(--accent)" }}
        >
          Begin
        </button>
      </div>

      {error && (
        <p className="text-[11px]" style={{ color: "var(--hot)" }}>
          {error}
        </p>
      )}

      <div className="space-y-2 pt-2">
        <div className="text-[10px] uppercase tracking-widest" style={{ color: "var(--muted)" }}>
          or raise one on
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => setText(p.build())}
              className="rounded-lg border p-3 text-left transition-colors hover:border-[var(--accent)]"
              style={{ borderColor: "var(--line)", background: "var(--panel)" }}
            >
              <div className="text-xs font-bold">{p.label}</div>
              <div className="sans mt-1 text-[11px] leading-snug" style={{ color: "var(--muted)" }}>
                {p.blurb}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function StageBar({ stage, loss }: { stage: ReturnType<typeof stageFor>; loss: number }) {
  const idx = Number.isFinite(loss) ? stageIndex(loss) : 0;
  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-3">
        <span className="text-lg font-bold tracking-widest" style={{ color: "var(--accent)" }}>
          {stage.label}
        </span>
        <span className="sans text-xs" style={{ color: "var(--muted)" }}>
          {stage.note}
        </span>
      </div>
      <div className="flex gap-1">
        {STAGES.map((_, i) => (
          <div
            key={i}
            className="h-0.5 flex-1 rounded-full transition-colors duration-500"
            style={{ background: i <= idx ? "var(--accent)" : "var(--line)" }}
          />
        ))}
      </div>
    </div>
  );
}

function Babble({ text, done }: { text: string; done: boolean }) {
  return (
    <div
      className="min-h-[132px] rounded-lg border p-4 text-[13px] leading-relaxed"
      style={{
        borderColor: done ? "var(--accent)" : "var(--line)",
        background: "var(--panel)",
        color: "var(--ink)",
      }}
    >
      {text ? (
        <span key={text.slice(0, 24)} className="arrive whitespace-pre-wrap break-words">
          {text}
        </span>
      ) : (
        <span className="pulse" style={{ color: "var(--muted)" }}>
          waking up…
        </span>
      )}
      {!done && text && <span className="caret" />}
    </div>
  );
}

function VitalsRow({ v, progress, phase }: { v: Vitals; progress: number; phase: Phase }) {
  const cells: [string, string][] = [
    ["steps", `${v.step}`],
    ["params", v.params.toLocaleString()],
    ["vocab", `${v.vocab} chars`],
    ["corpus", `${(v.chars / 1000).toFixed(1)}k`],
    ["backend", v.backend],
    ["steps/s", v.stepsPerSec.toFixed(1)],
  ];
  return (
    <div className="space-y-2">
      <div className="h-0.5 w-full overflow-hidden rounded-full" style={{ background: "var(--line)" }}>
        <div
          className="h-full transition-[width] duration-200"
          style={{ width: `${Math.min(100, progress * 100)}%`, background: "var(--accent)" }}
        />
      </div>
      <div className="grid grid-cols-3 gap-x-4 gap-y-2 sm:grid-cols-6">
        {cells.map(([k, val]) => (
          <div key={k}>
            <div className="text-[9px] uppercase tracking-widest" style={{ color: "var(--muted)" }}>
              {k}
            </div>
            <div className="text-[11px] tabular-nums">{val}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Done({
  firstWords,
  vitals,
  onReset,
  onKeepGoing,
}: {
  firstWords: string;
  vitals: Vitals | null;
  onReset: () => void;
  onKeepGoing: () => void;
}) {
  const [copied, setCopied] = useState(false);

  async function share() {
    const quote = firstWords.trim().slice(0, 140);
    const url = `${window.location.origin}/?w=${encodeURIComponent(quote)}&l=${
      vitals ? vitals.loss.toFixed(2) : ""
    }`;
    const msg = `I raised a neural network in my browser. Its first words were: "${quote}"`;
    if (navigator.share) {
      try {
        await navigator.share({ title: "FIRST WORDS", text: msg, url });
        return;
      } catch {
        /* dismissed — fall through to clipboard */
      }
    }
    await navigator.clipboard.writeText(`${msg} ${url}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <button
          onClick={share}
          className="rounded px-5 py-2.5 text-xs font-bold uppercase tracking-widest text-black"
          style={{ background: "var(--accent)" }}
        >
          {copied ? "copied" : "share its first words"}
        </button>
        <button
          onClick={onKeepGoing}
          className="rounded border px-5 py-2.5 text-xs uppercase tracking-widest transition-colors hover:bg-white/5"
          style={{ borderColor: "var(--accent)", color: "var(--accent)" }}
        >
          keep going
        </button>
        <button
          onClick={onReset}
          className="rounded border px-5 py-2.5 text-xs uppercase tracking-widest transition-colors hover:bg-white/5"
          style={{ borderColor: "var(--line)", color: "var(--muted)" }}
        >
          start over
        </button>
      </div>
      <p className="sans text-[11px]" style={{ color: "var(--muted)" }}>
        It stopped because the clock ran out, not because it finished. Another
        45 seconds will measurably improve it — the curve has not flattened.
      </p>
      <p className="sans text-[11px] leading-relaxed" style={{ color: "var(--muted)" }}>
        Sharing puts that one sampled sentence in the link. Your source text
        never leaves this tab, but the sentence the model invented from it does —
        so read it before you post it.
      </p>
    </div>
  );
}

function Footer() {
  return (
    <footer className="sans mt-auto pt-6 text-[11px] leading-relaxed" style={{ color: "var(--muted)" }}>
      An ~86k-parameter character model, trained from scratch in your browser on
      WebGL. It is roughly a millionth the size of the model that wrote this page,
      and it learns in the same direction: frequencies first, then spelling, then
      grammar, then something like a voice. Watching it is the closest you can get
      to seeing that happen at human speed.
    </footer>
  );
}

/**
 * Pick the most quotable of several samples.
 *
 * Prefers a candidate trimmed to whole words that contains real sentence
 * punctuation, because the share card is one line of text and a fragment ending
 * mid-word reads as broken rather than as charmingly infant.
 */
function pickQuotable(candidates: string[]): string {
  const cleaned = candidates.map(tidy).filter((s) => s.length > 24);
  if (cleaned.length === 0) return tidy(candidates[0] ?? "");
  return cleaned.sort((a, b) => score(b) - score(a))[0];
}

function tidy(s: string): string {
  const t = s.replace(/\s+/g, " ").trim();
  // drop the leading partial word — sampling starts mid-corpus
  const from = t.indexOf(" ");
  const body = from > 0 && from < 14 ? t.slice(from + 1) : t;
  // and the trailing partial word
  const last = body.lastIndexOf(" ");
  return (last > 24 ? body.slice(0, last) : body).trim();
}

function score(s: string): number {
  const words = s.split(" ").filter(Boolean);
  const real = words.filter((w) => /^[a-z]{2,12}[.,!?']?$/i.test(w)).length;
  return real / Math.max(1, words.length) + (/[.!?]/.test(s) ? 0.15 : 0);
}
