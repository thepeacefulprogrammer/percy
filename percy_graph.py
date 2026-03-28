"""Lightweight Neo4j helper for Percy's second-brain graph."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from neo4j import GraphDatabase


DEFAULT_URI = os.environ.get("PERCY_GRAPH_URI", "bolt://localhost:7687")
DEFAULT_USER = os.environ.get("PERCY_GRAPH_USER", "neo4j")
DEFAULT_PASSWORD = os.environ.get("PERCY_GRAPH_PASSWORD", "percybrain")
DEFAULT_DATABASE = os.environ.get("PERCY_GRAPH_DATABASE")

_driver = None


def get_driver():
    """Lazily create and cache the Neo4j driver."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            DEFAULT_URI,
            auth=(DEFAULT_USER, DEFAULT_PASSWORD),
        )
    return _driver


def _counters_to_dict(counters) -> Dict[str, int]:
    return {
        "nodes_created": counters.nodes_created,
        "nodes_deleted": counters.nodes_deleted,
        "relationships_created": counters.relationships_created,
        "relationships_deleted": counters.relationships_deleted,
        "properties_set": counters.properties_set,
        "labels_added": counters.labels_added,
        "labels_removed": counters.labels_removed,
        "indexes_added": counters.indexes_added,
        "indexes_removed": counters.indexes_removed,
        "constraints_added": counters.constraints_added,
        "constraints_removed": counters.constraints_removed,
        "system_updates": counters.system_updates,
    }


def run_cypher(
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    write: bool = False,
    database: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a Cypher query and return data + optional summary."""
    params = params or {}
    db = database or DEFAULT_DATABASE
    driver = get_driver()

    def _execute(tx):
        result = tx.run(cypher, **params)
        data = result.data()
        summary = result.consume()
        return {
            "data": data,
            "summary": {
                "database": summary.database,
                "counters": _counters_to_dict(summary.counters),
            },
        }

    with driver.session(database=db) as session:
        if write:
            return session.execute_write(_execute)
        return session.execute_read(_execute)


def query_graph(
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    database: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a read query (returns data + summary)."""
    return run_cypher(cypher, params, write=False, database=database)


def write_graph(
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    database: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a write query (returns data + summary)."""
    return run_cypher(cypher, params, write=True, database=database)


def to_json(result: Dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)

