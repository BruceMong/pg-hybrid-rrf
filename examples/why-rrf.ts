/**
 * The measurement that condemns `0.7 * cosine + 0.3 * ts_rank_cd`.
 *
 *   npx tsx examples/why-rrf.ts
 *
 * No database, no network: the two candidate lists below are shaped like real
 * ones — cosine similarity clustered tightly near the top, ts_rank_cd unbounded
 * and small. That difference in SPAN, not the constants, is what decides the
 * ranking.
 */

import { effectiveWeights, reciprocalRankFusion, weightedSumFusion } from "../src/index.js";
import type { RankedList } from "../src/index.js";

const vector: RankedList = {
  name: "vector",
  weight: 0.7,
  results: [
    { id: "rent-clause", score: 0.81 },
    { id: "rent-schedule", score: 0.79 },
    { id: "deposit-clause", score: 0.77 },
    { id: "notice-period", score: 0.74 },
    { id: "late-payment-penalty", score: 0.62 },
  ],
};

const fts: RankedList = {
  name: "fts",
  weight: 0.3,
  results: [
    { id: "late-payment-penalty", score: 0.098 },
    { id: "notice-period", score: 0.071 },
    { id: "indexation-clause", score: 0.044 },
    { id: "rent-clause", score: 0.021 },
  ],
};

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

console.log("What the weights actually do\n");
const influence = effectiveWeights([vector, fts]);
for (const [name, i] of Object.entries(influence)) {
  console.log(
    `  ${name.padEnd(7)} nominal ${pct(i.nominal).padStart(6)}` +
      `   span ${i.span.toFixed(3)}   effective ${pct(i.effective).padStart(6)}`,
  );
}
console.log(
  "\n  The keyword signal is configured at 30% and decides " +
    `${pct(influence.fts.effective)} of the ranking.`,
);
console.log("  Tuning the constants cannot fix this: the spans are set by the scoring");
console.log("  functions, and ts_rank_cd's span moves with document and query length.\n");

const table = (rows: Array<{ id: string; score: number }>) =>
  rows.map((r, i) => `  ${i + 1}. ${r.id.padEnd(22)} ${r.score.toFixed(4)}`).join("\n");

console.log("Weighted sum (0.7 / 0.3)\n");
console.log(table(weightedSumFusion([vector, fts])));

console.log("\nReciprocal Rank Fusion (k = 60, same 0.7 / 0.3)\n");
console.log(table(reciprocalRankFusion([vector, fts])));

const summed = weightedSumFusion([vector, fts]);
const fused = reciprocalRankFusion([vector, fts]);
const at = (id: string, list: Array<{ id: string }>) => list.findIndex((r) => r.id === id) + 1;
const ordinal = (n: number) => {
  const suffix = n % 10 === 1 && n % 100 !== 11 ? "st"
    : n % 10 === 2 && n % 100 !== 12 ? "nd"
    : n % 10 === 3 && n % 100 !== 13 ? "rd"
    : "th";
  return `${n}${suffix}`;
};
const ratioToTop = (id: string, list: Array<{ id: string; score: number }>) =>
  (list[0].score / list.find((r) => r.id === id)!.score).toFixed(1);

console.log(
  `\n  "late-payment-penalty" is the keyword signal's number one hit.` +
    `\n  Weighted sum puts it ${ordinal(at("late-payment-penalty", summed))}, below two documents the keyword signal` +
    `\n  never returned at all. RRF puts it ${ordinal(at("late-payment-penalty", fused))}.`,
);
console.log(
  `\n  "indexation-clause" is found by keywords only, so it has no cosine score.` +
    `\n  Both methods rank it last — but look at the distance from the top:` +
    `\n    weighted sum: ${ratioToTop("indexation-clause", summed)}x below the first result, i.e. out of the running` +
    `\n    RRF:          ${ratioToTop("indexation-clause", fused)}x below, i.e. still a candidate` +
    `\n  A missing signal costs you a rank under RRF. Under a weighted sum it costs` +
    `\n  you the whole term, and no weight can compensate for a zero.`,
);
