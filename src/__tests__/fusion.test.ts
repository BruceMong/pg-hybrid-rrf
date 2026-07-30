import { describe, expect, it } from "vitest";
import {
  effectiveWeights,
  reciprocalRankFusion,
  weightedSumFusion,
  type RankedList,
} from "../fusion.js";

const list = (name: string, entries: Array<[string, number]>, weight?: number): RankedList => ({
  name,
  weight,
  results: entries.map(([id, score]) => ({ id, score })),
});

/**
 * Score distributions measured on a real corpus, rounded: cosine similarity
 * clusters tightly near the top, `ts_rank_cd` is unbounded and small.
 */
const VECTOR = list("vector", [
  ["a", 0.81],
  ["b", 0.79],
  ["c", 0.77],
  ["d", 0.74],
  ["e", 0.62],
]);
const FTS = list("fts", [
  ["e", 0.098],
  ["d", 0.071],
  ["f", 0.044],
  ["a", 0.021],
]);

describe("reciprocalRankFusion", () => {
  it("fuses ranks, not scores", () => {
    const fused = reciprocalRankFusion([VECTOR, FTS], { k: 60 });
    // "a" is rank 1 for vector and rank 4 for fts.
    expect(fused[0].id).toBe("a");
    expect(fused[0].ranks).toEqual({ vector: 1, fts: 4 });
    expect(fused[0].score).toBeCloseTo(1 / 61 + 1 / 64, 10);
  });

  it("rewards a document both signals returned", () => {
    const fused = reciprocalRankFusion([VECTOR, FTS], { k: 60 });
    const byId = Object.fromEntries(fused.map((f) => [f.id, f]));
    // "d": vector 4 + fts 2. "b": vector 2 only. Agreement wins over a better
    // single-signal rank — which is the point of hybrid search.
    expect(byId.d.score).toBeGreaterThan(byId.b.score);
  });

  it("keeps documents only one signal found", () => {
    // Dropping them would make hybrid search strictly worse than either half.
    const ids = reciprocalRankFusion([VECTOR, FTS]).map((f) => f.id);
    expect(ids).toContain("f");
    expect(ids).toContain("b");
  });

  it("is immune to the scale of the incoming scores", () => {
    const scaled: RankedList = {
      name: "fts",
      results: FTS.results.map((r) => ({ ...r, score: r.score * 10_000 })),
    };
    expect(reciprocalRankFusion([VECTOR, scaled])).toEqual(
      reciprocalRankFusion([VECTOR, FTS]),
    );
  });

  it("re-sorts a list that arrives out of order", () => {
    const shuffled: RankedList = { name: "vector", results: [...VECTOR.results].reverse() };
    expect(reciprocalRankFusion([shuffled])[0].id).toBe("a");
  });

  it("honours weights as real multipliers", () => {
    const vectorOnly = reciprocalRankFusion([{ ...VECTOR, weight: 10 }, FTS]);
    // With the vector signal weighted 10x, its rank-1 document wins outright.
    expect(vectorOnly[0].id).toBe("a");
    const ftsOnly = reciprocalRankFusion([VECTOR, { ...FTS, weight: 10 }]);
    expect(ftsOnly[0].id).toBe("e");
  });

  it("lets k control how much the top ranks dominate", () => {
    // Small k: rank 1 vs rank 2 is a large gap. Large k: it flattens towards
    // counting how many signals returned the document.
    const tight = reciprocalRankFusion([VECTOR, FTS], { k: 1 });
    const flat = reciprocalRankFusion([VECTOR, FTS], { k: 1000 });
    const spread = (r: ReturnType<typeof reciprocalRankFusion>) => r[0].score / r[1].score;
    expect(spread(tight)).toBeGreaterThan(spread(flat));
  });

  it("breaks ties deterministically", () => {
    const one = list("a", [["x", 1]]);
    const two = list("b", [["y", 1]]);
    expect(reciprocalRankFusion([one, two]).map((f) => f.id)).toEqual(["x", "y"]);
  });

  it("truncates to limit and handles empty input", () => {
    expect(reciprocalRankFusion([VECTOR, FTS], { limit: 2 })).toHaveLength(2);
    expect(reciprocalRankFusion([])).toEqual([]);
    expect(reciprocalRankFusion([list("vector", [])])).toEqual([]);
  });
});

describe("effectiveWeights", () => {
  it("shows that a nominal 70/30 is nothing of the sort", () => {
    // This is the measurement that condemns `0.7*cosine + 0.3*ts_rank_cd`:
    // the keyword term is nominally 30% of the score and moves the ranking by
    // a few percent, because its span is an order of magnitude smaller.
    const influence = effectiveWeights([
      { ...VECTOR, weight: 0.7 },
      { ...FTS, weight: 0.3 },
    ]);

    expect(influence.vector.nominal).toBe(0.7);
    expect(influence.fts.nominal).toBe(0.3);
    expect(influence.fts.effective).toBeLessThan(0.15);
    expect(influence.vector.effective).toBeGreaterThan(0.85);
  });

  it("reports the span each signal actually covers", () => {
    const influence = effectiveWeights([VECTOR, FTS]);
    expect(influence.vector.span).toBeCloseTo(0.81 - 0.62, 10);
    expect(influence.fts.span).toBeCloseTo(0.098 - 0.021, 10);
  });

  it("agrees with the nominal weights when the scales do match", () => {
    // Two signals on the same 0..1 span: only then does a weighted sum mean
    // what it says.
    const influence = effectiveWeights([
      list("a", [["x", 1], ["y", 0]], 0.7),
      list("b", [["x", 1], ["y", 0]], 0.3),
    ]);
    expect(influence.a.effective).toBeCloseTo(0.7, 10);
    expect(influence.b.effective).toBeCloseTo(0.3, 10);
  });

  it("does not divide by zero on a flat or empty signal", () => {
    expect(effectiveWeights([list("flat", [["x", 0.5], ["y", 0.5]])]).flat.effective).toBe(0);
    expect(effectiveWeights([list("empty", [])]).empty).toMatchObject({ span: 0, effective: 0 });
  });
});

describe("weightedSumFusion", () => {
  it("is dominated by the wider-scaled signal, whatever the weights say", () => {
    const summed = weightedSumFusion([
      { ...VECTOR, weight: 0.7 },
      { ...FTS, weight: 0.3 },
    ]);
    const position = (id: string, r: { id: string }[]) => r.findIndex((x) => x.id === id);

    // "e" is the keyword signal's number one hit. Under the weighted sum it
    // ranks BELOW "b" and "c", which the keyword signal never returned at all:
    // 0.3 x 0.098 cannot make up for a cosine 0.15 lower. That is the 30% that
    // is not 30%.
    expect(position("e", summed)).toBeGreaterThan(position("b", summed));
    expect(position("e", summed)).toBeGreaterThan(position("c", summed));

    // Under RRF, being one signal's best hit beats being another's mid-ranker.
    const fused = reciprocalRankFusion([VECTOR, FTS]);
    expect(position("e", fused)).toBeLessThan(position("b", fused));
    expect(position("e", fused)).toBeLessThan(position("c", fused));
  });
});
