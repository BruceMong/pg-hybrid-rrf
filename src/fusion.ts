/**
 * Fusing several ranked result lists into one.
 *
 * THE BUG THIS FILE EXISTS FOR. Hybrid search is usually combined like this:
 *
 *     0.7 * cosine_similarity + 0.3 * ts_rank_cd
 *
 * It looks like "70% semantic, 30% keyword". It is not. Cosine similarity is
 * bounded in [0, 1] and, on real embeddings, clusters tightly — the top 50
 * candidates might all sit between 0.62 and 0.81. `ts_rank_cd` is unbounded and
 * on short documents typically lands around 0.01 to 0.1.
 *
 * So the semantic term moves the total by ~0.13 across the candidate set while
 * the keyword term moves it by ~0.03. The real split is closer to 80/20 the
 * other way round from what the constants say — and it drifts with document
 * length, corpus size and query length, so it cannot be fixed by tuning the
 * weights. Adding two numbers on incompatible scales is the error; the weights
 * were never the knob they appeared to be.
 *
 * Reciprocal Rank Fusion sidesteps it by combining RANKS, which have the same
 * scale by construction: 1/(k + rank). `effectiveWeights` below measures the
 * problem on your own data, so you do not have to take this on faith.
 */

export interface ScoredCandidate {
  id: string;
  /** Whatever your engine returned. Only the ORDER matters to RRF. */
  score: number;
}

export interface RankedList {
  /** Identifies the signal in the output: "vector", "fts", "recency"… */
  name: string;
  /**
   * Relative influence. Default 1. Under RRF this is a real multiplier on the
   * rank contribution, not a nominal weight on an unknown scale.
   */
  weight?: number;
  /** Candidates, best first. Re-sorted defensively by descending score. */
  results: ScoredCandidate[];
}

export interface FusedResult {
  id: string;
  /** Fused score. Comparable within one call only — never across queries. */
  score: number;
  /** 1-based rank in each list that returned this id. Absent means not returned. */
  ranks: Record<string, number>;
}

export interface FusionOptions {
  /**
   * RRF smoothing constant. 60 is the value from the original TREC paper and the
   * de-facto standard; keep it unless you have a reason.
   *
   * Small k makes the top ranks dominate (rank 1 vs 2 matters a lot). Large k
   * flattens towards counting how many lists returned the document at all.
   */
  k?: number;
  /** Truncate the fused list. */
  limit?: number;
}

const DEFAULT_K = 60;

function ordered(list: RankedList): ScoredCandidate[] {
  return [...list.results].sort((a, b) => b.score - a.score);
}

/**
 * Reciprocal Rank Fusion: `score = Σ weight / (k + rank)`.
 *
 * Scale-free by construction, so a keyword score of 0.03 and a cosine of 0.81
 * contribute on equal terms. A document returned by several signals accumulates
 * their contributions, which is the behaviour you want from hybrid search: a
 * result both signals agree on beats one that only the stronger signal liked.
 */
export function reciprocalRankFusion(
  lists: readonly RankedList[],
  options: FusionOptions = {},
): FusedResult[] {
  const k = options.k ?? DEFAULT_K;
  const fused = new Map<string, FusedResult>();

  for (const list of lists) {
    const weight = list.weight ?? 1;
    ordered(list).forEach((candidate, i) => {
      const rank = i + 1;
      const entry = fused.get(candidate.id) ?? { id: candidate.id, score: 0, ranks: {} };
      entry.score += weight / (k + rank);
      entry.ranks[list.name] = rank;
      fused.set(candidate.id, entry);
    });
  }

  const out = [...fused.values()].sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  return options.limit === undefined ? out : out.slice(0, options.limit);
}

/**
 * Weighted sum of raw scores — the approach RRF replaces.
 *
 * Provided so you can measure the difference on your own corpus, and to make a
 * migration reviewable. It is correct ONLY when every signal is bounded on the
 * same scale; see `effectiveWeights` before trusting it.
 */
export function weightedSumFusion(
  lists: readonly RankedList[],
  options: Pick<FusionOptions, "limit"> = {},
): FusedResult[] {
  const fused = new Map<string, FusedResult>();

  for (const list of lists) {
    const weight = list.weight ?? 1;
    ordered(list).forEach((candidate, i) => {
      const entry = fused.get(candidate.id) ?? { id: candidate.id, score: 0, ranks: {} };
      entry.score += weight * candidate.score;
      entry.ranks[list.name] = i + 1;
      fused.set(candidate.id, entry);
    });
  }

  const out = [...fused.values()].sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  return options.limit === undefined ? out : out.slice(0, options.limit);
}

export interface SignalInfluence {
  /** The weight you configured. */
  nominal: number;
  /** Its actual share of the ranking, 0..1. What you thought `nominal` meant. */
  effective: number;
  /** Score span of this signal across the candidate set (max - min). */
  span: number;
  min: number;
  max: number;
}

/**
 * How much each signal ACTUALLY moves a weighted-sum ranking.
 *
 * A weight only matters through the range it can move the total: a signal
 * spanning 0.03 with weight 0.3 moves the sum by 0.009, while one spanning 0.19
 * with weight 0.7 moves it by 0.133 — a 94/6 split from constants that read
 * 70/30. Run this on a real candidate set before believing any weighted sum.
 *
 * Report `effective`, not the constants, when someone asks how your hybrid
 * search is balanced. Under RRF the question does not arise: ranks are already
 * on one scale, so nominal weights mean what they say.
 */
export function effectiveWeights(
  lists: readonly RankedList[],
): Record<string, SignalInfluence> {
  const influences: Record<string, SignalInfluence> = {};
  let total = 0;

  for (const list of lists) {
    const weight = list.weight ?? 1;
    const scores = list.results.map((r) => r.score);
    const min = scores.length > 0 ? Math.min(...scores) : 0;
    const max = scores.length > 0 ? Math.max(...scores) : 0;
    const span = max - min;
    const contribution = Math.abs(weight) * span;
    total += contribution;
    influences[list.name] = { nominal: weight, effective: 0, span, min, max };
  }

  for (const list of lists) {
    const influence = influences[list.name];
    const contribution = Math.abs(influence.nominal) * influence.span;
    influence.effective = total > 0 ? contribution / total : 0;
  }

  return influences;
}
