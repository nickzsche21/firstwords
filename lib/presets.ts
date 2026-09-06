/**
 * Built-in corpora so the page is playable in one click.
 *
 * These are generated combinatorially rather than shipped as flat text: a few
 * hundred fragments expand into ~60KB of varied prose, which is enough signal
 * for the model to find a register without the repo carrying a corpus. They are
 * also all written here rather than borrowed, so nothing copyrighted is baked
 * into a page anyone can redistribute.
 */

export type Preset = {
  id: string;
  label: string;
  blurb: string;
  build: () => string;
};

function shuffleJoin(lines: string[], count: number, sep = "\n"): string {
  const out: string[] = [];
  for (let i = 0; i < count; i++) {
    out.push(lines[Math.floor(Math.random() * lines.length)]);
  }
  return out.join(sep);
}

const cross = (...pools: string[][]): string[] => {
  let acc: string[] = [""];
  for (const pool of pools) {
    const next: string[] = [];
    for (const a of acc) for (const b of pool) next.push(a + b);
    acc = next;
  }
  return acc;
};

/* ---------------------------------------------------------------- */

const ASSISTANT = cross(
  [
    "Great question! ",
    "I'd be happy to help with that. ",
    "Absolutely — ",
    "That's a really thoughtful point. ",
    "Let me break this down for you. ",
    "Certainly! ",
    "I appreciate you clarifying. ",
  ],
  [
    "It's important to note that ",
    "There are a few things to consider here. ",
    "At a high level, ",
    "Broadly speaking, ",
    "In many cases, ",
    "It really depends on your specific situation, but ",
  ],
  [
    "this approach has both advantages and drawbacks. ",
    "the answer varies depending on context. ",
    "reasonable people disagree on this. ",
    "there is no single correct answer. ",
    "the tradeoffs are worth weighing carefully. ",
    "I want to be careful not to overstate my confidence here. ",
  ],
  [
    "Would you like me to go deeper on any of these? ",
    "Let me know if you'd like me to elaborate! ",
    "I hope that helps! ",
    "Is there anything else I can assist you with today? ",
    "Feel free to ask follow-up questions. ",
    "Let me know if this is the direction you had in mind. ",
  ]
);

const LINKEDIN = cross(
  [
    "I got rejected 47 times before this. ",
    "My mentor once told me something I'll never forget. ",
    "Yesterday I fired my highest performer. ",
    "A candidate showed up to the interview 20 minutes late. ",
    "I turned down a seven figure offer last week. ",
    "Someone asked me why I still take out the trash myself. ",
  ],
  [
    "Here's what nobody tells you. ",
    "Most founders get this backwards. ",
    "It changed how I think about leadership forever. ",
    "The room went silent. ",
    "I'll never forget what happened next. ",
  ],
  [
    "Culture isn't ping pong tables. ",
    "Hire slow, fire fast, but hire for slope not intercept. ",
    "Your network is your net worth. ",
    "Talent is equally distributed. Opportunity is not. ",
    "Everyone wants the harvest. Nobody wants to plant. ",
  ],
  ["\n\nAgree? \n\n", "\n\nThoughts? \n\n", "\n\nRepost if this resonated. \n\n"]
);

const TAKES = cross(
  [
    "hot take: ",
    "unpopular opinion but ",
    "nobody is talking about how ",
    "genuinely think ",
    "the more I sit with it the more I believe ",
    "controversial but ",
  ],
  [
    "most software is downstream of one person's bad afternoon. ",
    "the best products are apologies for something. ",
    "taste is just pattern matching with better PR. ",
    "every abstraction is a bet about what will change. ",
    "shipping is a personality trait, not a skill. ",
    "the tooling discourse is a coping mechanism. ",
  ],
  [
    "anyway. ",
    "idk. thinking out loud. ",
    "will probably delete this. ",
    "no notes. ",
    "this is my whole thesis. ",
  ]
);

export const PRESETS: Preset[] = [
  {
    id: "assistant",
    label: "A helpful assistant",
    blurb: "Raise it on the register I was trained into. It gets unsettling.",
    build: () => shuffleJoin(ASSISTANT, 620, "\n"),
  },
  {
    id: "linkedin",
    label: "LinkedIn",
    blurb: "Broetry. The model learns the line break before it learns the words.",
    build: () => shuffleJoin(LINKEDIN, 520, "\n"),
  },
  {
    id: "takes",
    label: "Timeline takes",
    blurb: "Lowercase certainty. Converges alarmingly fast.",
    build: () => shuffleJoin(TAKES, 700, "\n"),
  },
];
