# pg-hybrid-rrf

Nearly every hybrid search on Postgres combines its two signals like this:

```sql
0.7 * (1 - (embedding <=> $1)) + 0.3 * ts_rank_cd(fts, query)
```

It reads as *70% semantic, 30% keyword*. It is not. Cosine similarity is bounded in `[0, 1]` and, on real embeddings, clusters tightly — the top 50 candidates might all sit between 0.62 and 0.81. `ts_rank_cd` is **unbounded** and on short chunks typically lands around 0.02 to 0.1.

So the semantic term moves the total by ~0.19 across the candidate set while the keyword term moves it by ~0.08. Measured on the numbers above:

```
  vector  nominal  70.0%   span 0.190   effective  85.2%
  fts     nominal  30.0%   span 0.077   effective  14.8%
```

**Tuning the constants cannot fix it.** The spans are set by the scoring functions, and `ts_rank_cd`'s span moves with document length, query length and corpus statistics. Adding two numbers on incompatible scales is the error; the weights were never the knob they appeared to be.

Reciprocal Rank Fusion combines **ranks**, which share a scale by construction:

```
score = Σ  weight / (k + rank)
```

This library gives you both the fusion and the Postgres query that produces the ranks — plus `effectiveWeights`, so you can run the measurement above on your own data instead of taking any of this on faith.

## Install

```sh
npm install pg-hybrid-rrf
```

Node ≥ 20, ESM, **no dependencies** — not even a Postgres driver. You inject the query function.

## Quickstart

```ts
import { createHybridSearch, buildIndexStatements } from "pg-hybrid-rrf";

const config = {
  table: "document_chunk",
  fts: { column: "fts", language: "french" },   // stored tsvector column
  extraColumns: ["document_id", "page_number"],
  filterColumns: ["tenant_id"],                 // scoped INSIDE each signal
  k: 60,
  candidateLimit: 50,
};

// Run once. Neither index is optional at scale.
for (const ddl of buildIndexStatements(config)) await pool.query(ddl);

const search = createHybridSearch(config, async (sql, params) => {
  const { rows } = await pool.query(sql, params);
  return rows;
});

const results = await search.search({
  embedding: queryVector,     // number[]
  text: "loyer impayé",       // parsed by websearch_to_tsquery
  filters: ["tenant-123"],
  limit: 10,
});
```

Each row carries `rrf_score` plus **both signals' raw scores and ranks** — `vector_score`, `fts_score`, `vector_rank`, `fts_rank`, `null` where that signal did not return the document. Keep them: they are what makes a bad result explainable instead of mysterious.

Works with `pg`, `postgres.js`, Drizzle's `execute`, a Supabase RPC — anything that runs parameterised SQL and returns rows. `search.explain(params)` hands you the SQL and parameters without running them.

## What the generated query does

One CTE per signal, `ROW_NUMBER()` for the rank, and a `FULL OUTER JOIN` to fuse:

- **Each signal is limited server-side** (`candidateLimit`), so you transfer 10 rows instead of two candidate sets, and one statement uses both the vector index and the GIN index.
- **`FULL OUTER JOIN`, not `INNER`.** Documents only one signal found are kept. Dropping them would make hybrid search strictly worse than either half alone.
- **Filters are applied inside each CTE.** This matters more than it looks: filtering after fusion lets a large tenant's documents fill the 50 candidate slots and starve the tenant you were searching for. Put your scope in `filterColumns`.
- **Identifiers are validated and double-quoted; values are always bound.** A bound parameter cannot carry a table name, so config identifiers are interpolated — and therefore checked against `^[A-Za-z_][A-Za-z0-9_$]*$` and rejected loudly if odd. The one exception is `fts.expression`, which is your SQL by construction: never build it from user input.

## Fusing in the application instead

If your signals do not both live in Postgres — a reranker, a recency boost, an external keyword engine — fuse the lists yourself:

```ts
import { reciprocalRankFusion } from "pg-hybrid-rrf";

const fused = reciprocalRankFusion([
  { name: "vector", results: vectorHits },        // [{ id, score }]
  { name: "fts", results: keywordHits },
  { name: "recency", weight: 0.5, results: recentHits },
], { k: 60, limit: 20 });
// → [{ id, score, ranks: { vector: 1, fts: 4 } }]
```

Any number of signals, and `weight` is a real multiplier here rather than a nominal share of an unknown scale.

## Measuring your own weights

Before trusting any weighted sum — including one you inherited:

```ts
import { effectiveWeights } from "pg-hybrid-rrf";

console.table(effectiveWeights([
  { name: "vector", weight: 0.7, results: vectorHits },
  { name: "fts", weight: 0.3, results: keywordHits },
]));
```

`effective` is the share of the ranking each signal actually decides, derived from `weight × span`. Report that number, not the constants, when someone asks how your hybrid search is balanced.

```sh
npx tsx examples/why-rrf.ts
```

Prints the measurement and both rankings side by side on realistic distributions. The keyword signal's top hit lands 5th under the weighted sum — below two documents the keyword signal never returned at all — and 3rd under RRF.

## Choosing k

`k = 60` comes from the original TREC paper and is the de-facto standard. Keep it unless you have a reason:

- **Smaller k** makes the top ranks dominate — rank 1 versus rank 2 becomes a large gap.
- **Larger k** flattens towards *counting how many signals returned the document at all*.

## Limits

- **RRF discards score magnitude, on purpose.** A document that is a 0.95 cosine match and one that is 0.70 are rank 1 and rank 2; the gap between them is gone. If your scores are genuinely calibrated and comparable, a weighted sum carries more information — verify that with `effectiveWeights` first, because it usually is not true.
- **Fused scores are not comparable across queries.** They depend on the candidate set. Do not threshold on them, do not store them as a relevance measure.
- **A kNN search has no cutoff.** The vector CTE always returns its top `candidateLimit`, however irrelevant — so "the vector signal did not find it" really means "it fell outside the candidate window". Sizing that window is a relevance decision, not just a performance one.
- **Recall is bounded by `candidateLimit` per signal.** A document ranked 60th by both signals cannot be rescued by fusion.
- **`ts_rank_cd` still needs a sensible text search configuration.** Use `french`, `english` — `simple` (the default here) does no stemming, so *loyers* will not match *loyer*.
- **Two signals only, in SQL.** More signals means fusing in the application with `reciprocalRankFusion`.
- **No embedding generation, no reranking, no query expansion.** This library fuses and queries; nothing else.

## API

| Export | Purpose |
| --- | --- |
| `createHybridSearch(config, query)` | `search(params)` and `explain(params)`. |
| `buildHybridSearchQuery(config, params)` | `{ sql, params }` — use it directly if you prefer. |
| `buildIndexStatements(config)` | The GIN / HNSW / filter index DDL, as a checklist. |
| `reciprocalRankFusion(lists, opts?)` | Fuse N ranked lists by rank. |
| `weightedSumFusion(lists, opts?)` | The approach RRF replaces, for A/B comparison. |
| `effectiveWeights(lists)` | What your weights actually decide. |
| `disagreements(rows, limit?)` | Rows the two signals disagree on most — where tuning starts. |
| `quoteIdentifier(name)` | Validate and quote an identifier. |

## Development

```sh
npm install
npm test                 # 35 unit tests, no database

npm run db:up            # Postgres 17 + pgvector on :55432
DATABASE_URL=postgres://postgres:postgres@localhost:55432/hybrid npm test
npm run db:down
```

The integration suite skips itself without `DATABASE_URL`. It is the only test that proves the generated SQL is *valid* SQL — the unit tests assert on strings, and a string assertion cannot tell you the planner accepts it. It earned its place immediately: it caught the text search configuration being emitted as an identifier (`"french"`) where Postgres wants a `regconfig` literal (`'french'`), which fails with `column "french" does not exist`. Every string assertion in the suite was perfectly happy.

## Provenance

Extracted from the retrieval layer of a production RAG system, where the migration off `0.7 * cosine + 0.3 * ts_rank_cd` is what surfaced the scale mismatch in the first place. `effectiveWeights` is that analysis turned into an API, so the next person does not have to rediscover it by reading score distributions in a psql session.

## License

MIT
