# FIRST WORDS

Paste any text. A neural network is **born in your browser tab** and raised on
it, live, while you watch.

It goes NOISE → SOUND → LETTERS → SYLLABLES → WORDS → PHRASES → VOICE. You see
the loss curve fall and the babble resolve into language in about a minute. When
the clock runs out, whatever it managed to say is its first words, and that's
what you share.

No API. No key. No server. **Your text never leaves the tab** — training runs on
your own GPU via WebGL, and the model is destroyed when you close the page.

## Why this exists

Everyone has now talked to a language model. Almost nobody has watched one
*learn*. The interesting part of these systems isn't the finished article, it's
the order in which competence arrives: character frequencies first, then
spelling, then grammar, then something that reads like a voice. That sequence is
completely invisible at the scale the famous models train at — months, thousands
of GPUs, behind an API.

At ~86k parameters it fits in one tab and takes 45 seconds, and the sequence is
exactly the same.

## Engineering notes

The three decisions that mattered, all of them forced by measurement rather than
taste:

**Training is time-budgeted, not step-budgeted.** Step throughput varies by more
than 10× across the machines this runs on. Benchmarked on the pure-JS CPU
fallback with no GPU at all, the original architecture managed 105 steps in 45
seconds — useless — while the shipped one manages 353. A fixed step count would
mean 40 seconds for some people and ten minutes for others. Spending a fixed
amount of *time* bounds the experience everywhere, and turns the step counter
into an honest readout of your own hardware. "Keep going" extends the same
model rather than restarting it.

**Sampling uses `dataSync()`, deliberately.** Generating 110 characters is 110
sequential single-value GPU readbacks. The async `data()` path resolves each by
polling on a timer, and browsers clamp timers in a background tab to ~1s — which
turned one sample into a two-minute freeze. A blocking read has no timer in the
loop and wins outright at this size.

**Yielding avoids `requestAnimationFrame` when hidden.** `tf.nextFrame()` is
rAF, which browsers stop firing entirely in a background tab, so a run left in
another tab froze mid-training. Visible tabs still yield per frame; hidden ones
fall through to a `MessageChannel` macrotask, which isn't throttled the way a
background `setTimeout` is.

Architecture: `embedding(24) → flatten → dense(256, relu) → dropout → softmax`
over a 12-character context, Adam at 5e-3, batch 64. An MLP rather than an RNN
because on tfjs/WebGL it runs an order of magnitude more steps per second, and
steps-per-second *is* the experience.

## Run it

```bash
npm install
npm run dev
```

http://localhost:3333

## Sharing

Finishing gives you `/?w=<first words>&l=<loss>`, and that route generates its
own OG image, so a pasted link unfurls with the sentence that particular model
invented. Every run produces a different one.

Your source text stays local. The one sampled sentence goes in the link — the UI
says so before you click.

Set `NEXT_PUBLIC_SITE_URL` to your deployed origin so absolute OG URLs resolve.
It's inlined at build time, so changing it needs a redeploy, not a restart.

## Stack

Next.js 16 · React 19 · TypeScript · Tailwind v4 · TensorFlow.js (WebGL).

tfjs is ~1MB and is loaded lazily on the first click, so the landing page
doesn't pay for it.

## License

MIT
