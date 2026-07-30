/**
 * Generating the hybrid search SQL.
 *
 * Fusing in Postgres rather than in the application matters once the corpus is
 * real: each signal is limited server-side, so you transfer 20 rows instead of
 * two candidate sets, and one statement can use both the vector index
 * (ivfflat/HNSW) and the GIN index.
 *
 * IDENTIFIER SAFETY. Table and column names come from your config, not from user
 * input, but they are still interpolated — a bound parameter cannot carry an
 * identifier. Every one goes through `quoteIdentifier`, which validates the shape
 * and double-quotes it. Values are always bound, never interpolated.
 */

export interface FtsColumnConfig {
  /** Stored `tsvector` column. The fast path: index it with GIN. */
  column: string;
  /** Text search configuration, e.g. `french`. Default `simple`. */
  language?: string;
}

export interface FtsExpressionConfig {
  /**
   * SQL expression to build the tsvector from, e.g. `coalesce(title,'') || ' ' || content`.
   *
   * TRUSTED INPUT: inserted verbatim, like a schema definition. Never build it
   * from anything a user typed.
   */
  expression: string;
  language?: string;
}

export interface HybridSearchConfig {
  /** Table holding the chunks. May be schema-qualified: `rag.chunks`. */
  table: string;
  /** Primary key, returned as `id`. Default `id`. */
  idColumn?: string;
  /** Chunk text column. Default `content`. */
  contentColumn?: string;
  /** `vector(n)` column. Default `embedding`. */
  embeddingColumn?: string;
  fts: FtsColumnConfig | FtsExpressionConfig;
  /** Extra columns to return: document id, page number, metadata. */
  extraColumns?: string[];
  /**
   * Equality-filtered columns, in the order their values are passed.
   *
   * A multi-tenant scope belongs HERE: the filter is applied inside each
   * signal's CTE, so the per-signal candidate limit is spent on this tenant's
   * rows. Filtering after fusion instead lets a large neighbour's documents fill
   * the candidate set and starve the tenant of results.
   */
  filterColumns?: string[];
  /** RRF constant. Default 60. */
  k?: number;
  /** Candidates each signal contributes before fusion. Default 50. */
  candidateLimit?: number;
  /** Weight of the vector signal. Default 1. */
  vectorWeight?: number;
  /** Weight of the full-text signal. Default 1. */
  ftsWeight?: number;
}

export interface BuiltQuery {
  sql: string;
  /** Positional parameters matching `$1…$n`. */
  params: unknown[];
}

export interface SearchParams {
  /** Query embedding, sent as a pgvector literal. */
  embedding: number[];
  /** Raw query text; `websearch_to_tsquery` parses it. */
  text: string;
  /** Rows after fusion. Default 10. */
  limit?: number;
  /** Values for `filterColumns`, in the same order. */
  filters?: unknown[];
}

const DEFAULTS = {
  idColumn: "id",
  contentColumn: "content",
  embeddingColumn: "embedding",
  k: 60,
  candidateLimit: 50,
  weight: 1,
  language: "simple",
  limit: 10,
};

const IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_$]*$/;

/**
 * Validate and double-quote an identifier, one dot-separated part at a time.
 *
 * Rejecting anything exotic rather than escaping it is deliberate: a config that
 * needs a quote character in a table name is a mistake far more often than a
 * requirement, and failing loudly beats generating SQL nobody reads.
 */
export function quoteIdentifier(identifier: string): string {
  const parts = identifier.split(".");
  for (const part of parts) {
    if (!IDENTIFIER_RE.test(part)) {
      throw new Error(
        `Invalid SQL identifier ${JSON.stringify(identifier)}: ` +
          `expected letters, digits and underscores, got ${JSON.stringify(part)}`,
      );
    }
  }
  return parts.map((p) => `"${p}"`).join(".");
}

/**
 * A text search configuration goes in as a `regconfig` LITERAL, not as an
 * identifier: `to_tsvector('french', …)`. Double-quoting it makes Postgres look
 * for a column of that name and fail with `column "french" does not exist`.
 *
 * Validated with the identifier rules first, so the literal cannot carry a quote.
 */
function regconfigLiteral(language: string): string {
  quoteIdentifier(language);
  return `'${language}'::regconfig`;
}

function tsvectorExpression(config: HybridSearchConfig, language: string): string {
  return "column" in config.fts
    ? quoteIdentifier(config.fts.column)
    : `to_tsvector(${regconfigLiteral(language)}, ${config.fts.expression})`;
}

/**
 * The hybrid query: one CTE per signal, fused by rank.
 *
 * `ROW_NUMBER()` gives each signal a dense 1..n rank and the fusion adds
 * `weight / (k + rank)`, so a keyword score of 0.03 and a cosine of 0.81 count on
 * equal terms. The `FULL OUTER JOIN` keeps documents only one signal found —
 * dropping them would make hybrid search strictly worse than either half alone.
 */
export function buildHybridSearchQuery(
  config: HybridSearchConfig,
  params: SearchParams,
): BuiltQuery {
  const table = quoteIdentifier(config.table);
  const id = quoteIdentifier(config.idColumn ?? DEFAULTS.idColumn);
  const content = quoteIdentifier(config.contentColumn ?? DEFAULTS.contentColumn);
  const embedding = quoteIdentifier(config.embeddingColumn ?? DEFAULTS.embeddingColumn);
  const language = config.fts.language ?? DEFAULTS.language;
  const tsvector = tsvectorExpression(config, language);
  const lang = regconfigLiteral(language);
  const extras = (config.extraColumns ?? []).map(quoteIdentifier);
  const filterColumns = (config.filterColumns ?? []).map(quoteIdentifier);

  const k = config.k ?? DEFAULTS.k;
  const candidateLimit = config.candidateLimit ?? DEFAULTS.candidateLimit;
  const vectorWeight = config.vectorWeight ?? DEFAULTS.weight;
  const ftsWeight = config.ftsWeight ?? DEFAULTS.weight;

  const filterValues = params.filters ?? [];
  if (filterValues.length !== filterColumns.length) {
    throw new Error(
      `Expected ${filterColumns.length} filter value(s) for ` +
        `[${(config.filterColumns ?? []).join(", ")}], got ${filterValues.length}`,
    );
  }

  // $1 embedding, $2 query text, then one per filter, then the limit. The filter
  // placeholders are referenced by BOTH CTEs, hence bound once and reused.
  const values: unknown[] = [`[${params.embedding.join(",")}]`, params.text];
  const filterSql = filterColumns
    .map((column, i) => {
      values.push(filterValues[i]);
      return `      AND ${column} = $${values.length}`;
    })
    .join("\n");
  values.push(params.limit ?? DEFAULTS.limit);
  const limitParam = `$${values.length}`;

  const cteProjection = extras.map((c) => `    ${c},`).join("\n");
  const joinProjection = extras
    .map((c) => `  COALESCE(v.${c}, f.${c}) AS ${c},`)
    .join("\n");
  const line = (s: string) => (s.length > 0 ? `${s}\n` : "");

  const sql = `WITH vector_candidates AS (
  SELECT
    ${id} AS id,
    ${content} AS content,
${line(cteProjection)}    1 - (${embedding} <=> $1::vector) AS vector_score,
    ROW_NUMBER() OVER (ORDER BY ${embedding} <=> $1::vector) AS vector_rank
  FROM ${table}
  WHERE ${embedding} IS NOT NULL
${line(filterSql)}  ORDER BY ${embedding} <=> $1::vector
  LIMIT ${candidateLimit}
),
fts_candidates AS (
  SELECT
    ${id} AS id,
    ${content} AS content,
${line(cteProjection)}    ts_rank_cd(${tsvector}, websearch_to_tsquery(${lang}, $2)) AS fts_score,
    ROW_NUMBER() OVER (
      ORDER BY ts_rank_cd(${tsvector}, websearch_to_tsquery(${lang}, $2)) DESC
    ) AS fts_rank
  FROM ${table}
  WHERE ${tsvector} @@ websearch_to_tsquery(${lang}, $2)
${line(filterSql)}  LIMIT ${candidateLimit}
)
SELECT
  COALESCE(v.id, f.id) AS id,
  COALESCE(v.content, f.content) AS content,
${line(joinProjection)}  v.vector_score,
  f.fts_score,
  v.vector_rank,
  f.fts_rank,
  COALESCE(${vectorWeight}::double precision / (${k} + v.vector_rank), 0)
    + COALESCE(${ftsWeight}::double precision / (${k} + f.fts_rank), 0) AS rrf_score
FROM vector_candidates v
FULL OUTER JOIN fts_candidates f ON f.id = v.id
ORDER BY rrf_score DESC
LIMIT ${limitParam}`;

  return { sql, params: values };
}

/**
 * Index DDL the query needs to be fast, as a checklist.
 *
 * Neither index is optional at scale: without the GIN index the full-text CTE
 * sequential-scans the table, and without a vector index the ORDER BY does too —
 * and a sequential scan behind a `LIMIT 50` is the usual reason "hybrid search is
 * slow" turns out to have nothing to do with the fusion.
 */
export function buildIndexStatements(
  config: HybridSearchConfig & { vectorIndex?: "hnsw" | "ivfflat"; lists?: number },
): string[] {
  const table = quoteIdentifier(config.table);
  const embedding = quoteIdentifier(config.embeddingColumn ?? DEFAULTS.embeddingColumn);
  const bare = config.table.split(".").pop() ?? config.table;
  const statements: string[] = [];

  if ("column" in config.fts) {
    const column = quoteIdentifier(config.fts.column);
    statements.push(
      `CREATE INDEX IF NOT EXISTS ${quoteIdentifier(`idx_${bare}_fts`)} ON ${table} USING GIN (${column});`,
    );
  }

  const kind = config.vectorIndex ?? "hnsw";
  statements.push(
    kind === "hnsw"
      ? `CREATE INDEX IF NOT EXISTS ${quoteIdentifier(`idx_${bare}_embedding`)} ON ${table} USING hnsw (${embedding} vector_cosine_ops);`
      : `CREATE INDEX IF NOT EXISTS ${quoteIdentifier(`idx_${bare}_embedding`)} ON ${table} USING ivfflat (${embedding} vector_cosine_ops) WITH (lists = ${config.lists ?? 100});`,
  );

  for (const column of config.filterColumns ?? []) {
    statements.push(
      `CREATE INDEX IF NOT EXISTS ${quoteIdentifier(`idx_${bare}_${column}`)} ON ${table} (${quoteIdentifier(column)});`,
    );
  }

  return statements;
}
