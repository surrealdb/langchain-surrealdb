import json
from typing import Any

from langchain_community.graphs.graph_document import GraphDocument, Node
from langchain_community.graphs.graph_store import GraphStore
from surrealdb import (
    AsyncHttpSurrealConnection,
    AsyncWsSurrealConnection,
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
    RecordID,
    Value,
)

SurrealConnection = BlockingWsSurrealConnection | BlockingHttpSurrealConnection
SurrealAsyncConnection = AsyncWsSurrealConnection | AsyncHttpSurrealConnection

CREATE_SOURCE_QUERY = """
    CREATE type::table($table)
    CONTENT $content
"""

CREATE_NODE_QUERY = """
    CREATE $record_id
    CONTENT $content
"""

RELATE_QUERY = """
    RELATE $in->(type::table($relation))->$out
    CONTENT $content
"""


class SurrealDBGraph(GraphStore):
    def __init__(
        self,
        connection: SurrealConnection,
        *,
        table_prefix: str = "graph_",
        relation_prefix: str = "relation_",
    ) -> None:
        super().__init__()
        self.connection = connection
        self.table_prefix = table_prefix
        self.relation_prefix = relation_prefix

    def _query(self, surql: str, vars: dict[str, Value]) -> dict[str, Any]:
        return self.connection.query_raw(surql, vars)

    def _build_node_recordid(self, node: Node) -> RecordID:
        return RecordID(self.table_prefix + node.type, node.id)

    @property
    def get_schema(self) -> str:
        """Return the schema of the Graph database"""
        info = self._query("INFO FOR DB", {})
        tables = info["result"][0]["result"]["tables"].keys()
        nodes = []
        edges = []
        for table in tables:
            if table.startswith(self.table_prefix):
                nodes.append(table)
            elif table.startswith(self.relation_prefix):
                edges.append(table)
        return json.dumps({"nodes": nodes, "edges": edges})

    @property
    def get_structured_schema(self) -> dict[str, Any]:
        """Return the schema of the Graph database"""
        raise NotImplementedError

    def query(self, query: str, params: dict[str, Value] = {}) -> list[dict[str, Any]]:
        """Query the graph."""
        res = self._query(query, params)
        if "error" in res:
            raise Exception(res["error"]["message"])
        else:
            return res["result"][0]["result"]

    def refresh_schema(self) -> None:
        """Refresh the graph schema information."""
        raise NotImplementedError

    def delete_nodes(
        self, ids: list[tuple[str, str | None]] | None = None, **kwargs: Any
    ) -> None:
        """Delete nodes (and relations) in the graph."""
        if ids is not None:
            for table, _id in ids:
                if _id is None:
                    _ = self.connection.delete(table)
                else:
                    _ = self.connection.delete(RecordID(table, _id))
        else:
            # find all tables
            info = self.connection.query("INFO FOR DB", {})
            if isinstance(info, dict):
                for table in info.get("tables", {}).keys():
                    _ = self.connection.delete(table)
            _ = self.connection.delete(self.table_prefix + "source")

    def add_graph_documents(
        self, graph_documents: list[GraphDocument], include_source: bool = False
    ) -> None:
        """Take GraphDocument as input and uses it to construct a graph."""
        for doc in graph_documents:
            source = None
            if include_source:
                source = self._query(
                    CREATE_SOURCE_QUERY,
                    {
                        "table": self.table_prefix + "source",
                        "content": {
                            "page_content": doc.source.page_content,
                            "metadata": doc.source.metadata,
                        },
                    },
                )
                source = source["result"][0]["result"][0]

            for node in doc.nodes:
                _ = self._query(
                    CREATE_NODE_QUERY,
                    {
                        "record_id": self._build_node_recordid(node),
                        "content": node.properties,
                    },
                )
                if include_source and source is not None:
                    _ = self._query(
                        RELATE_QUERY,
                        {
                            "in": source["id"],
                            "relation": "MENTIONS",
                            "out": self._build_node_recordid(node),
                            "content": {},
                        },
                    )

            for rel in doc.relationships:
                _ = self._query(
                    RELATE_QUERY,
                    {
                        "in": self._build_node_recordid(rel.source),
                        "relation": self.relation_prefix + rel.type,
                        "out": self._build_node_recordid(rel.target),
                        "content": rel.properties,
                    },
                )
