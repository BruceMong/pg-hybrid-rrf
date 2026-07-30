import { describe, expect, it } from "vitest";
import {
  buildHybridSearchQuery,
  buildIndexStatements,
  quoteIdentifier,
  type HybridSearchConfig,
} from "../sql.js";
import { createHybridSearch, disagreements, type HybridSearchRow } from "../search.js";

const CONFIG: HybridSearchConfig = {
  table: "document_chunk",
  fts: { column: "fts", language: "french" },
  extraColumns: ["document_id", "page_number"],
  filterColumns: ["tenant_id"],
};

const PARAMS = { embedding: [0.1, 0.2, 0.3], text: "loyer impayé", filters: ["t-1"], limit: 5 };

describe("quoteIdentifier", () => {
  it("quotes each part of a qualified name", () => {
    expect(quoteIdentifier("chunks")).toBe('"chunks"');
    expect(quoteIdentifier("rag.chunks")).toBe('"rag"."chunks"');
  });

  it("rejects anything that is not a plain identifier", () => {
    // An identifier cannot be a bound parameter, so it is interpolated — which
    // makes validation the only line of defence.
    expect(() => quoteIdentifier('chunks"; DROP TABLE users; --')).toThrow(/Invalid SQL identifier/);
    expect(() => quoteIdentifier("2fast")).toThrow();
    expect(() => quoteIdentifier("")).toThrow();
    expect(() => quoteIdentifier("a b")).toThrow();
  });
});

describe("buildHybridSearchQuery", () => {
  it("binds values and interpolates only quoted identifiers", () => {
    const { sql, params } = buildHybridSearchQuery(CONFIG, PARAMS);

    expect(params).toEqual(["[0.1,0.2,0.3]", "loyer impayé", "t-1", 5]);
    expect(sql).toContain('"document_chunk"');
    // No value from params appears in the SQL text.
    expect(sql).not.toContain("loyer");
    expect(sql).not.toContain("t-1");
  });

  it("fuses by rank, not by score", () => {
    const { sql } = buildHybridSearchQuery(CONFIG, PARAMS);
    expect(sql).toContain("ROW_NUMBER() OVER");
    expect(sql).toMatch(/1::double precision \/ \(60 \+ v\.vector_rank\)/);
    expect(sql).toMatch(/1::double precision \/ \(60 \+ f\.fts_rank\)/);
    // The scores are returned for inspection but never summed.
    expect(sql).not.toMatch(/vector_score\s*\*/);
  });

  it("keeps rows only one signal found", () => {
    expect(buildHybridSearchQuery(CONFIG, PARAMS).sql).toContain("FULL OUTER JOIN");
  });

  it("applies the filter inside each signal, not after fusion", () => {
    // Filtering after fusion lets a large neighbour's rows fill the candidate set
    // and starve this tenant of results.
    const { sql } = buildHybridSearchQuery(CONFIG, PARAMS);
    const occurrences = sql.match(/AND "tenant_id" = \$3/g);
    expect(occurrences).toHaveLength(2);
    const [vectorCte, rest] = sql.split("fts_candidates AS (");
    expect(vectorCte).toContain('AND "tenant_id" = $3');
    expect(rest).toContain('AND "tenant_id" = $3');
  });

  it("reuses one placeholder per filter across both signals", () => {
    const { params } = buildHybridSearchQuery(CONFIG, PARAMS);
    expect(params.filter((p) => p === "t-1")).toHaveLength(1);
  });

  it("refuses a filter value count that does not match the columns", () => {
    expect(() => buildHybridSearchQuery(CONFIG, { ...PARAMS, filters: [] })).toThrow(
      /Expected 1 filter value/,
    );
  });

  it("projects the extra columns through the join", () => {
    const { sql } = buildHybridSearchQuery(CONFIG, PARAMS);
    expect(sql).toContain('COALESCE(v."page_number", f."page_number") AS "page_number"');
  });

  it("works with no extras and no filters", () => {
    const { sql, params } = buildHybridSearchQuery(
      { table: "chunks", fts: { column: "fts" } },
      { embedding: [1], text: "x" },
    );
    expect(params).toEqual(["[1]", "x", 10]);
    expect(sql).not.toContain("AND ");
    expect(sql).toContain("FULL OUTER JOIN");
  });

  it("builds the tsvector on the fly when there is no stored column", () => {
    const { sql } = buildHybridSearchQuery(
      {
        table: "chunks",
        fts: { expression: "coalesce(title,'') || ' ' || content", language: "french" },
      },
      { embedding: [1], text: "x" },
    );
    expect(sql).toContain(`to_tsvector('french'::regconfig, coalesce(title,'') || ' ' || content)`);
  });

  it("passes the text search configuration as a regconfig literal", () => {
    // Regression, caught only by a real database: double-quoting it makes
    // Postgres look for a column and fail with `column "french" does not exist`.
    const { sql } = buildHybridSearchQuery(CONFIG, PARAMS);
    expect(sql).toContain("websearch_to_tsquery('french'::regconfig, $2)");
    expect(sql).not.toContain('websearch_to_tsquery("french"');
  });

  it("refuses a text search configuration that could break out of the literal", () => {
    expect(() =>
      buildHybridSearchQuery({ ...CONFIG, fts: { column: "fts", language: "fr', 'x" } }, PARAMS),
    ).toThrow(/Invalid SQL identifier/);
  });

  it("carries the configured k and weights into the fusion", () => {
    const { sql } = buildHybridSearchQuery(
      { ...CONFIG, k: 20, vectorWeight: 2, ftsWeight: 0.5 },
      PARAMS,
    );
    expect(sql).toContain("2::double precision / (20 + v.vector_rank)");
    expect(sql).toContain("0.5::double precision / (20 + f.fts_rank)");
  });

  it("rejects an injected identifier from the config", () => {
    expect(() =>
      buildHybridSearchQuery({ ...CONFIG, table: 'chunks" WHERE 1=1 --' }, PARAMS),
    ).toThrow(/Invalid SQL identifier/);
  });
});

describe("buildIndexStatements", () => {
  it("covers both signals and the filters", () => {
    const statements = buildIndexStatements(CONFIG);
    expect(statements.some((s) => s.includes("USING GIN"))).toBe(true);
    expect(statements.some((s) => s.includes("USING hnsw"))).toBe(true);
    expect(statements.some((s) => s.includes('("tenant_id")'))).toBe(true);
  });

  it("offers ivfflat with a lists parameter", () => {
    const statements = buildIndexStatements({ ...CONFIG, vectorIndex: "ivfflat", lists: 200 });
    expect(statements.some((s) => s.includes("USING ivfflat") && s.includes("lists = 200"))).toBe(
      true,
    );
  });

  it("skips the GIN index when the tsvector is computed at query time", () => {
    const statements = buildIndexStatements({
      table: "chunks",
      fts: { expression: "content" },
    });
    expect(statements.some((s) => s.includes("USING GIN"))).toBe(false);
  });
});

describe("createHybridSearch", () => {
  it("hands the built SQL to the injected executor", async () => {
    const calls: Array<{ sql: string; params: unknown[] }> = [];
    const search = createHybridSearch(CONFIG, async (sql, params) => {
      calls.push({ sql, params });
      return [];
    });

    await search.search(PARAMS);
    expect(calls).toHaveLength(1);
    expect(calls[0].params).toEqual(["[0.1,0.2,0.3]", "loyer impayé", "t-1", 5]);
  });

  it("can explain without running anything", () => {
    const search = createHybridSearch(CONFIG, async () => {
      throw new Error("must not run");
    });
    expect(search.explain(PARAMS).sql).toContain("FULL OUTER JOIN");
  });
});

describe("disagreements", () => {
  it("surfaces the rows the two signals disagree on most", () => {
    const row = (id: string, vector_rank: number | null, fts_rank: number | null) =>
      ({ id, content: "", vector_score: null, fts_score: null, vector_rank, fts_rank, rrf_score: 0 }) as HybridSearchRow;

    const ordered = disagreements([row("agree", 1, 2), row("clash", 2, null), row("close", 3, 3)], 50);
    expect(ordered.map((r) => r.id)).toEqual(["clash", "agree", "close"]);
  });
});
