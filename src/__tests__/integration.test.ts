/**
 * End to end against a real Postgres with pgvector.
 *
 *   npm run db:up
 *   DATABASE_URL=postgres://postgres:postgres@localhost:55432/hybrid npm test
 *
 * Skipped without DATABASE_URL, so the unit suite stays offline. This is the only
 * test that proves the generated SQL is valid SQL — everything else asserts on
 * strings, and a string assertion cannot tell you the planner accepts it.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Client } from "pg";
import { buildHybridSearchQuery, buildIndexStatements } from "../sql.js";
import { createHybridSearch, type HybridSearchRow } from "../search.js";

const url = process.env.DATABASE_URL;
const suite = url ? describe : describe.skip;

/** 3 dimensions is enough to place documents in a plane we can reason about. */
const CONFIG = {
  table: "chunks",
  fts: { column: "fts", language: "english" as const },
  extraColumns: ["document_id"],
  filterColumns: ["tenant_id"],
  candidateLimit: 20,
};

const ROWS: Array<[string, string, string, [number, number, number]]> = [
  ["c1", "t1", "The tenant must pay the monthly rent on the fifth", [1, 0, 0]],
  ["c2", "t1", "Rent is due at the beginning of each month", [0.9, 0.1, 0]],
  ["c3", "t1", "The deposit equals two months of rent", [0.8, 0.2, 0]],
  ["c4", "t1", "Termination requires three months notice", [0, 1, 0]],
  ["c5", "t1", "A completely unrelated paragraph about bicycles", [0, 0, 1]],
  // Another tenant's rows, deliberately the best keyword matches: if the filter
  // is applied after fusion instead of inside each signal, they show up.
  ["x1", "t2", "rent rent rent monthly rent payment rent", [1, 0, 0]],
  ["x2", "t2", "monthly rent rent rent rent due rent", [1, 0, 0]],
];

suite("hybrid search against Postgres", () => {
  let client: Client;

  beforeAll(async () => {
    client = new Client({ connectionString: url });
    await client.connect();
    await client.query("CREATE EXTENSION IF NOT EXISTS vector");
    await client.query("DROP TABLE IF EXISTS chunks");
    await client.query(`
      CREATE TABLE chunks (
        id text PRIMARY KEY,
        tenant_id text NOT NULL,
        document_id text NOT NULL,
        content text NOT NULL,
        embedding vector(3),
        fts tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
      )
    `);
    for (const [id, tenant, content, embedding] of ROWS) {
      await client.query(
        "INSERT INTO chunks (id, tenant_id, document_id, content, embedding) VALUES ($1, $2, $3, $4, $5)",
        [id, tenant, `doc-${id}`, content, `[${embedding.join(",")}]`],
      );
    }
    for (const statement of buildIndexStatements(CONFIG)) {
      await client.query(statement);
    }
  }, 60_000);

  afterAll(async () => {
    await client?.end();
  });

  const search = () =>
    createHybridSearch(
      CONFIG,
      async (sql, params) => (await client.query(sql, params)).rows as HybridSearchRow[],
    );

  it("runs the generated SQL and returns both signals' scores", async () => {
    const rows = await search().search({
      embedding: [1, 0, 0],
      text: "monthly rent",
      filters: ["t1"],
      limit: 5,
    });

    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0]).toHaveProperty("rrf_score");
    expect(rows[0]).toHaveProperty("document_id");
    // Ordered by the fused score, descending.
    const scores = rows.map((r) => Number(r.rrf_score));
    expect([...scores].sort((a, b) => b - a)).toEqual(scores);
  });

  it("never leaks another tenant's rows, even when they match best", async () => {
    const rows = await search().search({
      embedding: [1, 0, 0],
      text: "monthly rent",
      filters: ["t1"],
      limit: 10,
    });
    expect(rows.map((r) => r.id)).not.toContain("x1");
    expect(rows.map((r) => r.id)).not.toContain("x2");
  });

  it("rescues a document that falls outside the vector candidate window", async () => {
    // A kNN search has no distance threshold: it always returns its top N,
    // however irrelevant. So "vector-only" and "fts-only" are about the CANDIDATE
    // WINDOW, not about relevance — with candidateLimit 2, the vector side only
    // sees the two nearest rows, and full-text is what rescues "bicycles".
    const narrow = createHybridSearch({ ...CONFIG, candidateLimit: 2 }, async (sql, params) =>
      (await client.query(sql, params)).rows as HybridSearchRow[],
    );
    const rows = await narrow.search({
      embedding: [1, 0, 0],
      text: "bicycles",
      filters: ["t1"],
      limit: 10,
    });

    const c5 = rows.find((r) => r.id === "c5");
    expect(c5).toBeDefined();
    expect(c5!.fts_rank).not.toBeNull();
    expect(c5!.vector_rank).toBeNull();
    // And it is kept in the output rather than dropped by the join.
    expect(Number(c5!.rrf_score)).toBeGreaterThan(0);
  });

  it("returns a document only the vector signal can find", async () => {
    const rows = await search().search({
      embedding: [0, 1, 0],
      text: "bicycles",
      filters: ["t1"],
      limit: 10,
    });
    const c4 = rows.find((r) => r.id === "c4");
    expect(c4).toBeDefined();
    expect(c4!.vector_rank).not.toBeNull();
    expect(c4!.fts_score).toBeNull();
  });

  it("survives a query text with no full-text match at all", async () => {
    const rows = await search().search({
      embedding: [1, 0, 0],
      text: "zzzzznomatch",
      filters: ["t1"],
      limit: 3,
    });
    // The vector side still answers; the fts side contributes nothing.
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((r) => r.fts_rank === null)).toBe(true);
  });

  it("is accepted by the planner as a single statement", async () => {
    const { sql, params } = buildHybridSearchQuery(CONFIG, {
      embedding: [1, 0, 0],
      text: "rent",
      filters: ["t1"],
    });
    const plan = await client.query(`EXPLAIN ${sql}`, params);
    expect(plan.rows.length).toBeGreaterThan(0);
  });
});
