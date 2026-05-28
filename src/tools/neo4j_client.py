"""Neo4j graph store — structure only, no embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from neo4j import GraphDatabase

from config.settings import settings


@dataclass
class GraphEntity:
    name: str
    type: str


@dataclass
class GraphRelation:
    source: str
    target: str
    relation: str
    chunk_id: str


class Neo4jGraphClient:
    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> Neo4jGraphClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def ensure_schema(self) -> None:
        queries = [
            "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
        ]
        with self._driver.session() as session:
            for q in queries:
                session.run(q)

    def upsert_document(self, name: str, source_path: str = "") -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (d:Document {name: $name})
                SET d.source_path = $source_path
                """,
                name=name,
                source_path=source_path,
            )

    def upsert_chunk(
        self,
        *,
        chunk_id: str,
        text: str,
        document_name: str,
        page: int,
        section: str = "",
        extraction_ok: bool = True,
    ) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (d:Document {name: $document_name})
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text,
                    c.page = $page,
                    c.section = $section,
                    c.extraction_ok = $extraction_ok
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                text=text,
                document_name=document_name,
                page=page,
                section=section,
                extraction_ok=extraction_ok,
            )

    def upsert_entity(self, entity: GraphEntity, chunk_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name})
                SET e.type = $type
                WITH e
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                name=entity.name,
                type=entity.type,
                chunk_id=chunk_id,
            )

    _ALLOWED_RELATIONS = frozenset({
        "SUBSIDIARY_OF", "PRODUCES", "HAS_RISK", "REPORTED", "LOCATED_IN",
        "COMPETES_WITH", "REGULATED_BY", "EMPLOYS", "INVESTS_IN",
        "PARTNERS_WITH", "RELATED_TO",
    })

    def upsert_relation(self, rel: GraphRelation) -> None:
        rel_type = rel.relation.upper()
        if rel_type not in self._ALLOWED_RELATIONS:
            rel_type = "RELATED_TO"

        query = f"""
        MATCH (a:Entity {{name: $source}})
        MATCH (b:Entity {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        SET r.sources = CASE
            WHEN $chunk_id IN coalesce(r.sources, []) THEN r.sources
            ELSE coalesce(r.sources, []) + $chunk_id
        END
        """
        with self._driver.session() as session:
            session.run(
                query,
                source=rel.source,
                target=rel.target,
                chunk_id=rel.chunk_id,
            )

    def load_extraction(
        self,
        *,
        chunk_id: str,
        text: str,
        document_name: str,
        page: int,
        section: str,
        entities: list[GraphEntity],
        relations: list[GraphRelation],
        extraction_ok: bool = True,
    ) -> None:
        self.upsert_chunk(
            chunk_id=chunk_id,
            text=text,
            document_name=document_name,
            page=page,
            section=section,
            extraction_ok=extraction_ok,
        )
        for entity in entities:
            self.upsert_entity(entity, chunk_id)
        for relation in relations:
            self.upsert_relation(relation)

    def chunk_extraction_done(self, chunk_id: str) -> bool:
        """True if chunk was successfully extracted (skip when --resume)."""
        with self._driver.session() as session:
            row = session.run(
                """
                OPTIONAL MATCH (c:Chunk {chunk_id: $chunk_id})
                OPTIONAL MATCH (c)-[:MENTIONS]->(e)
                WITH c, count(e) AS mention_count
                RETURN CASE
                    WHEN c IS NULL THEN false
                    WHEN c.extraction_ok = true THEN true
                    WHEN c.extraction_ok = false THEN false
                    WHEN mention_count > 0 THEN true
                    ELSE false
                END AS done
                """,
                chunk_id=chunk_id,
            ).single()
        if row is None:
            return False
        return bool(row["done"])

    @staticmethod
    def _records(result) -> list[dict]:
        """Materialize a Neo4j result before the session closes."""
        return [dict(r) for r in result]

    def search_entities(self, query: str, limit: int = 8) -> list[dict]:
        """Fuzzy match entity names related to query tokens."""
        tokens = [t for t in re.split(r"[\s，、]+", query) if len(t) >= 2][:6]
        with self._driver.session() as session:
            if tokens:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $q
                       OR any(tok IN $tokens WHERE e.name CONTAINS tok)
                    RETURN DISTINCT e.name AS name, e.type AS type
                    LIMIT $limit
                    """,
                    q=query.strip()[:80],
                    tokens=tokens,
                    limit=limit,
                )
            else:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name CONTAINS $q
                    RETURN DISTINCT e.name AS name, e.type AS type
                    LIMIT $limit
                    """,
                    q=query.strip()[:80],
                    limit=limit,
                )
            rows = self._records(result)
        return [{"name": r["name"], "type": r["type"]} for r in rows]

    def get_graph_context(
        self,
        entity_names: list[str],
        *,
        max_relations: int = 20,
        max_chunks: int = 5,
    ) -> tuple[str, list[str]]:
        """Build text context from entity neighborhood and linked chunks."""
        if not entity_names:
            return "", []

        with self._driver.session() as session:
            relations = self._records(
                session.run(
                    """
                    UNWIND $names AS name
                    MATCH (e:Entity {name: name})-[r]->(o:Entity)
                    WHERE type(r) <> 'MENTIONS'
                    RETURN e.name AS src, type(r) AS rel, o.name AS tgt,
                           coalesce(r.sources, []) AS sources
                    LIMIT $max_rel
                    """,
                    names=entity_names[:10],
                    max_rel=max_relations,
                )
            )
            chunks = self._records(
                session.run(
                    """
                    UNWIND $names AS name
                    MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: name})
                    RETURN DISTINCT c.chunk_id AS chunk_id, c.page AS page,
                           left(c.text, 500) AS text
                    LIMIT $max_chunks
                    """,
                    names=entity_names[:10],
                    max_chunks=max_chunks,
                )
            )

        lines: list[str] = ["【知识图谱】"]
        chunk_ids: list[str] = []
        for r in relations:
            lines.append(f"- ({r['src']})-[:{r['rel']}]->({r['tgt']})")
        if chunks:
            lines.append("\n【关联文档片段】")
            for c in chunks:
                chunk_ids.append(c["chunk_id"])
                lines.append(
                    f"[{c['chunk_id']}] (p{c['page']}) {c['text']}"
                )
        return "\n".join(lines), chunk_ids

    def search_chunks_by_keywords(
        self,
        keywords: list[str],
        *,
        boost_phrases: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict]:
        """Full-text search on Chunk.text with optional phrase boosting."""
        if not keywords:
            return []
        boost_phrases = boost_phrases or []
        with self._driver.session() as session:
            rows = self._records(
                session.run(
                    """
                    MATCH (c:Chunk)
                    WHERE any(kw IN $keywords WHERE c.text CONTAINS kw)
                       OR any(bp IN $boost WHERE c.text CONTAINS bp)
                    WITH c,
                         size([kw IN $keywords WHERE c.text CONTAINS kw]) AS hit_count,
                         reduce(s = 0, bp IN $boost |
                           s + CASE WHEN c.text CONTAINS bp THEN
                             CASE bp
                               WHEN '发行人子公司' THEN 12
                               WHEN '系发行人的全资子公司' THEN 10
                               WHEN '全资子公司' THEN 8
                               WHEN '实际控制人' THEN 10
                               WHEN '控股股东' THEN 8
                               WHEN '王兴兴' THEN 10
                               ELSE 4
                             END
                           ELSE 0 END
                         ) AS boost_score
                    RETURN c.chunk_id AS chunk_id, c.page AS page, c.text AS text,
                           hit_count, boost_score,
                           hit_count + boost_score AS score
                    ORDER BY score DESC, c.page ASC
                    LIMIT $limit
                    """,
                    keywords=keywords[:12],
                    boost=boost_phrases[:10],
                    limit=limit,
                )
            )
        return rows

    def search_subsidiary_chunks(self, *, limit: int = 6) -> list[dict]:
        """Prioritize the issuer subsidiary listing section (around p56–58)."""
        return self.search_chunks_by_keywords(
            [
                "发行人子公司",
                "全资子公司",
                "宇树机器人",
                "宁波宇树",
                "重庆宇羿",
                "上海高羿",
            ],
            boost_phrases=[
                "发行人子公司",
                "系发行人的全资子公司",
                "全资子公司",
            ],
            limit=limit,
        )

    def search_controller_chunks(self, *, limit: int = 4) -> list[dict]:
        """Prioritize actual controller / shareholder disclosure."""
        return self.search_chunks_by_keywords(
            ["实际控制人", "控股股东", "王兴兴", "表决权"],
            boost_phrases=["实际控制人", "控股股东", "王兴兴"],
            limit=limit,
        )

    def search_risk_chunks(self, *, limit: int = 6) -> list[dict]:
        """Prioritize prospectus risk-factor section."""
        return self.search_chunks_by_keywords(
            [
                "风险因素",
                "可能面临",
                "经营业绩波动",
                "研发不及预期",
                "存货跌价",
                "应收账款",
            ],
            boost_phrases=["风险因素", "第二节 风险因素", "风险"],
            limit=limit,
        )

    def search_subsidiary_relations(self, parent_hint: str = "宇树", limit: int = 15) -> list[dict]:
        """Entity–entity SUBSIDIARY_OF edges involving the issuer."""
        with self._driver.session() as session:
            rows = self._records(
                session.run(
                    """
                    MATCH (a:Entity)-[r:SUBSIDIARY_OF]->(b:Entity)
                    WHERE a.name CONTAINS $hint OR b.name CONTAINS $hint
                    RETURN a.name AS src, b.name AS tgt, coalesce(r.sources, []) AS sources
                    LIMIT $limit
                    """,
                    hint=parent_hint,
                    limit=limit,
                )
            )
            rows += self._records(
                session.run(
                    """
                    MATCH (a:Entity)<-[r:SUBSIDIARY_OF]-(b:Entity)
                    WHERE a.name CONTAINS $hint OR b.name CONTAINS $hint
                    RETURN b.name AS src, a.name AS tgt, coalesce(r.sources, []) AS sources
                    LIMIT $limit
                    """,
                    hint=parent_hint,
                    limit=limit,
                )
            )
        return rows

    def expand_entities_from_chunk_ids(self, chunk_ids: list[str], limit: int = 10) -> list[str]:
        if not chunk_ids:
            return []
        with self._driver.session() as session:
            rows = self._records(
                session.run(
                    """
                    UNWIND $ids AS cid
                    MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity)
                    RETURN DISTINCT e.name AS name
                    LIMIT $limit
                    """,
                    ids=chunk_ids,
                    limit=limit,
                )
            )
        return [r["name"] for r in rows]

    def stats(self) -> dict[str, int]:
        with self._driver.session() as session:
            row = session.run(
                """
                OPTIONAL MATCH (d:Document) WITH count(d) AS documents
                OPTIONAL MATCH (c:Chunk) WITH documents, count(c) AS chunks
                OPTIONAL MATCH (e:Entity) WITH documents, chunks, count(e) AS entities
                OPTIONAL MATCH ()-[r]->() WHERE type(r) <> 'HAS_CHUNK' AND type(r) <> 'MENTIONS'
                RETURN documents, chunks, entities, count(r) AS relations
                """
            ).single()
        return {
            "documents": row["documents"],
            "chunks": row["chunks"],
            "entities": row["entities"],
            "relations": row["relations"],
        }
