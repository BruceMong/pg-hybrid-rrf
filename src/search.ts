/**
 * Running the query.
 *
 * The database client is injected, so this package depends on no driver: `pg`,
 * `postgres.js`, Drizzle's `execute`, a Supabase RPC — anything that can run
 * parameterised SQL and hand back rows.
 */

import { buildHybridSearchQuery, type HybridSearchConfig, type SearchParams } from "./sql.js";

export interface HybridSearchRow {
  id: string;
  content: string;
  /** Cosine similarity, 0..1. `null` when only the full-text signal found it. */
  vector_score: number | null;
  /** `ts_rank_cd`, unbounded. `null` when only the vector signal found it. */
  fts_score: number | null;
  /** 1-based rank within the vector candidates. */
  vector_rank: number | null;
  fts_rank: number | null;
  /** The fused score rows are ordered by. Comparable within one query only. */
  rrf_score: number;
  [column: string]: unknown;
}

/** Runs parameterised SQL and returns rows. */
export type QueryExecutor = (
  sql: string,
  params: unknown[],
) => Promise<HybridSearchRow[]>;

export interface HybridSearch {
  search: (params: SearchParams) => Promise<HybridSearchRow[]>;
  /** The SQL and parameters, without running them — for logging or EXPLAIN. */
  explain: (params: SearchParams) => { sql: string; params: unknown[] };
}

export function createHybridSearch(
  config: HybridSearchConfig,
  query: QueryExecutor,
): HybridSearch {
  return {
    async search(params) {
      const { sql, params: values } = buildHybridSearchQuery(config, params);
      return query(sql, values);
    },
    explain(params) {
      return buildHybridSearchQuery(config, params);
    },
  };
}

/**
 * Rows where the two signals disagree most — the ones worth reading when tuning.
 *
 * A document the vector signal ranked 2nd and the keyword signal did not return
 * at all tells you more about your setup than the ten results both agreed on.
 * Sorted by the size of the disagreement, missing ranks counted as
 * `candidateLimit + 1`.
 */
export function disagreements(
  rows: readonly HybridSearchRow[],
  candidateLimit = 50,
): HybridSearchRow[] {
  const rank = (r: number | null) => r ?? candidateLimit + 1;
  return [...rows].sort(
    (a, b) =>
      Math.abs(rank(b.vector_rank) - rank(b.fts_rank)) -
      Math.abs(rank(a.vector_rank) - rank(a.fts_rank)),
  );
}
