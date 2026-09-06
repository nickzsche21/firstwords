/**
 * Developmental stages, keyed off smoothed cross-entropy in nats/character.
 *
 * The thresholds are empirical for a char-level model on English-ish text:
 * a uniform distribution over ~90 characters sits near 4.5, unigram frequency
 * alone gets you to roughly 3.0, and anything under 1.5 means the model has
 * started reproducing multi-word structure rather than plausible spelling.
 *
 * They are the whole narrative — a loss number means nothing to most people,
 * but "it has found the alphabet" does.
 */

export type Stage = {
  id: string;
  label: string;
  note: string;
  /** entered once smoothed loss drops below this */
  below: number;
};

export const STAGES: Stage[] = [
  { id: "voice", label: "VOICE", note: "It is doing an impression of you.", below: 1.42 },
  { id: "phrases", label: "PHRASES", note: "It has found your grammar. Not your meaning.", below: 1.68 },
  { id: "words", label: "WORDS", note: "Real words. Wrong order. This is the funny part.", below: 1.98 },
  { id: "syllables", label: "SYLLABLES", note: "Pronounceable nonsense. It sounds like a language now.", below: 2.32 },
  { id: "letters", label: "LETTERS", note: "It has learned which letters are common. Nothing else.", below: 2.72 },
  { id: "sound", label: "SOUND", note: "It found the alphabet. It has no idea what to do with it.", below: 3.3 },
  { id: "noise", label: "NOISE", note: "It does not yet know that characters exist.", below: Infinity },
];

export function stageFor(loss: number): Stage {
  // STAGES is ordered most-developed first, so the first match is the best one
  // the model currently qualifies for.
  return STAGES.find((s) => loss < s.below) ?? STAGES[STAGES.length - 1];
}

export function stageIndex(loss: number): number {
  const s = stageFor(loss);
  // count from the bottom so higher = more developed, for progress bars
  return STAGES.length - 1 - STAGES.indexOf(s);
}
