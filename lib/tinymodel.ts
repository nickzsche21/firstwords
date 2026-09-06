/**
 * A character-level language model small enough to be born, raised and buried
 * inside one browser tab.
 *
 * Deliberately not an RNN. On tfjs/WebGL an MLP over a fixed context window
 * runs an order of magnitude more steps per second than a GRU of comparable
 * quality, and steps-per-second is the entire experience here — the point is
 * watching the babble resolve into language in real time, not squeezing out
 * the last few nats of loss.
 */

import * as tf from "@tensorflow/tfjs";

export const CONTEXT = 12;
const MAX_VOCAB = 112;
const MAX_CHARS = 400_000;

export type Vocab = {
  chars: string[];
  stoi: Map<string, number>;
  size: number;
};

/**
 * Frequency-capped character vocabulary. Rare characters (a stray emoji, one
 * Cyrillic name in an English chat log) collapse into a single slot rather than
 * each buying a full row of the embedding and output matrices.
 */
export function buildVocab(text: string): Vocab {
  const freq = new Map<string, number>();
  for (const ch of text) freq.set(ch, (freq.get(ch) ?? 0) + 1);

  const ranked = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_VOCAB - 1)
    .map(([ch]) => ch)
    .sort();

  const chars = [" ", ...ranked.filter((c) => c !== " ")];
  const stoi = new Map(chars.map((c, i) => [c, i]));
  return { chars, stoi, size: chars.length };
}

export function encode(text: string, v: Vocab): Int32Array {
  const out = new Int32Array(text.length);
  let i = 0;
  for (const ch of text) out[i++] = v.stoi.get(ch) ?? 0;
  return out.slice(0, i);
}

export function decode(ids: ArrayLike<number>, v: Vocab): string {
  let s = "";
  for (let i = 0; i < ids.length; i++) s += v.chars[ids[i]] ?? "";
  return s;
}

export function prepareCorpus(raw: string): string {
  // Normalise newlines and collapse the runs of blank lines that chat exports
  // are full of; they otherwise dominate the training signal.
  const t = raw
    .replace(new RegExp("\\r\\n?", "g"), "\n")
    .replace(/\n{3,}/g, "\n\n");
  return t.length > MAX_CHARS ? t.slice(0, MAX_CHARS) : t;
}

export type TrainTick = {
  step: number;
  /** 0 → 1 through the current training budget */
  progress: number;
  loss: number;
  /** exponential moving average — the raw per-batch loss is far too noisy to plot */
  smoothed: number;
  sample?: string;
};

export type TrainOptions = {
  /**
   * Wall-clock budget, not a step count.
   *
   * Step throughput varies by more than an order of magnitude across the
   * machines this runs on — a discrete GPU on WebGL versus tfjs's pure-JS CPU
   * fallback — so a fixed step count means a 40-second wait for some people and
   * a ten-minute one for others. Spending a fixed amount of time and letting
   * the resulting model be as good as that machine allows keeps the experience
   * bounded everywhere, and the step counter becomes an honest readout of how
   * fast your hardware actually is.
   */
  durationMs: number;
  batchSize: number;
  /** non-zero when continuing an already-trained model */
  startStep?: number;
  onTick: (t: TrainTick) => void;
  shouldStop: () => boolean;
};

/**
 * How often to draw a sample.
 *
 * Sampling is the single most expensive thing in the loop — it is N sequential
 * forward passes, each with a GPU readback that the training step does not need
 * — so it cannot run often. Early steps are where the output changes fastest
 * and are worth the cost; past that the interval widens sharply.
 */
function sampleInterval(step: number): number {
  return step < 150 ? 15 : 45;
}

export class TinyModel {
  readonly vocab: Vocab;
  readonly params: number;
  /** carried across successive train() calls so "keep going" resumes the curve */
  lastLoss = 0;
  private model: tf.LayersModel;
  private data: Int32Array;

  constructor(vocab: Vocab, data: Int32Array) {
    this.vocab = vocab;
    this.data = data;

    const model = tf.sequential();
    model.add(
      tf.layers.embedding({
        inputDim: vocab.size,
        outputDim: 24,
        inputLength: CONTEXT,
      })
    );
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 256, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.15 }));
    model.add(tf.layers.dense({ units: vocab.size, activation: "softmax" }));

    model.compile({
      optimizer: tf.train.adam(5e-3),
      loss: "sparseCategoricalCrossentropy",
    });

    this.model = model;
    this.params = model.countParams();
  }

  /** Random windows into the corpus. Cheaper and less correlated than epochs. */
  private batch(size: number): [tf.Tensor2D, tf.Tensor1D] {
    const xs = new Int32Array(size * CONTEXT);
    const ys = new Int32Array(size);
    const limit = Math.max(1, this.data.length - CONTEXT - 1);
    for (let b = 0; b < size; b++) {
      const i = Math.floor(Math.random() * limit);
      xs.set(this.data.subarray(i, i + CONTEXT), b * CONTEXT);
      ys[b] = this.data[i + CONTEXT];
    }
    // float32, not int32: tfjs layers run tf.floor internally on both the
    // embedding input and the sparse-CCE labels, and that op rejects int32.
    return [
      tf.tensor2d(Float32Array.from(xs), [size, CONTEXT], "float32"),
      tf.tensor1d(Float32Array.from(ys), "float32"),
    ];
  }

  async train(opts: TrainOptions): Promise<number> {
    const { durationMs, batchSize, onTick, shouldStop } = opts;
    let step = opts.startStep ?? 0;
    let smoothed = this.lastLoss;

    const t0 = performance.now();
    let elapsed = 0;

    while (elapsed < durationMs) {
      if (shouldStop()) break;
      step++;

      const [xs, ys] = this.batch(batchSize);
      const res = await this.model.trainOnBatch(xs, ys);
      xs.dispose();
      ys.dispose();

      const loss = Array.isArray(res) ? (res[0] as number) : (res as number);
      smoothed = smoothed === 0 ? loss : smoothed * 0.94 + loss * 0.06;
      this.lastLoss = smoothed;

      elapsed = performance.now() - t0;
      const tick: TrainTick = {
        step,
        progress: Math.min(1, elapsed / durationMs),
        loss,
        smoothed,
      };
      if (step % sampleInterval(step) === 0) {
        tick.sample = this.generate(110, 0.7);
      }
      onTick(tick);

      // Hand control back so the loss curve and the babble actually paint.
      await yieldToBrowser();
    }

    return step;
  }

  /**
   * Sample forward one character at a time.
   *
   * Seeded from a real position in the corpus so early output shows the model
   * failing to *continue something real*, which reads far better than watching
   * it fail to continue a blank.
   */
  generate(length: number, temperature: number): string {
    const start = Math.floor(
      Math.random() * Math.max(1, this.data.length - CONTEXT)
    );
    const ctx = Array.from(this.data.subarray(start, start + CONTEXT));
    while (ctx.length < CONTEXT) ctx.unshift(0);

    const out: number[] = [];
    for (let i = 0; i < length; i++) {
      // dataSync, deliberately, despite the GPU pipeline stall.
      //
      // Sampling is 100+ sequential single-character reads. The async data()
      // path resolves each one by polling on a timer, and a browser clamps
      // timers in a background tab to ~1s — which turned a single sample into a
      // two-minute freeze. A blocking read costs microseconds and has no timer
      // in the loop at all; for tensors this small it wins outright.
      const probs = tf.tidy(() => {
        const input = tf.tensor2d([ctx.slice(-CONTEXT)], [1, CONTEXT], "float32");
        const logits = this.model.predict(input) as tf.Tensor2D;
        return logits.dataSync() as Float32Array;
      });
      const next = sampleFrom(probs, temperature);
      out.push(next);
      ctx.push(next);
    }
    return decode(out, this.vocab);
  }

  dispose() {
    this.model.dispose();
  }
}

/**
 * Temperature sampling over an already-softmaxed distribution.
 *
 * Rescaling happens in log space: doing it directly as p**(1/t) underflows to
 * an all-zero distribution at low temperatures, and the cumulative walk below
 * would then return index 0 forever — a model that appears to only ever emit
 * spaces.
 */
function sampleFrom(probs: Float32Array, temperature: number): number {
  const t = Math.max(0.05, temperature);
  const scaled = new Float32Array(probs.length);
  let total = 0;
  for (let i = 0; i < probs.length; i++) {
    const p = Math.exp(Math.log(Math.max(probs[i], 1e-9)) / t);
    scaled[i] = p;
    total += p;
  }
  let r = Math.random() * total;
  for (let i = 0; i < scaled.length; i++) {
    r -= scaled[i];
    if (r <= 0) return i;
  }
  return scaled.length - 1;
}

/**
 * Yield between training steps.
 *
 * tf.nextFrame() is requestAnimationFrame, which browsers stop firing entirely
 * in a background tab — a run left in another tab would freeze mid-training and
 * silently resume on return. Visible tabs still yield per frame so painting
 * stays smooth; hidden ones fall through to a MessageChannel macrotask, which
 * is not throttled the way a background setTimeout is (clamped to ~1s).
 */
function yieldToBrowser(): Promise<void> {
  if (typeof document === "undefined" || document.visibilityState === "visible") {
    return tf.nextFrame();
  }
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => {
      ch.port1.close();
      resolve();
    };
    ch.port2.postMessage(null);
  });
}

export async function initBackend(): Promise<string> {
  try {
    await tf.setBackend("webgl");
    await tf.ready();
    return "webgl";
  } catch {
    await tf.setBackend("cpu");
    await tf.ready();
    return "cpu";
  }
}
