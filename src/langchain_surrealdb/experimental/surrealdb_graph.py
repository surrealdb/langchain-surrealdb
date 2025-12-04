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
from typing_extensions import cast, override

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
        self.connection: SurrealConnection = connection
        self.table_prefix: str = table_prefix
        self.relation_prefix: str = relation_prefix

    def _query(self, surql: str, vars: dict[str, Value]) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        return self.connection.query_raw(surql, vars)

    def _build_node_recordid(self, node: Node) -> RecordID:
        return RecordID(self.table_prefix + node.type, node.id)

    @property
    @override
    def get_schema(self) -> str:
        """Return the schema of the Graph database"""
        info = self._query("INFO FOR DB", {})
        nodes: list[str] = []
        edges: list[str] = []
        # tables = info["result"][0]["result"]["tables"].keys()
        temp = info["result"]  # pyright: ignore[reportAny]
        if (
            isinstance(temp, list)
            and len(temp) > 0  # pyright: ignore[reportUnknownArgumentType]
            and isinstance(temp[0], dict)
            and "result" in temp[0]
        ):
            temp = temp[0]["result"]  # pyright: ignore[reportUnknownVariableType]
        if (
            isinstance(temp, dict)
            and "tables" in temp
            and isinstance(temp["tables"], dict)
        ):
            for table in temp["tables"].keys():  # pyright: ignore[reportUnknownVariableType]
                assert isinstance(table, str)
                if table.startswith(self.table_prefix):
                    nodes.append(table)
                elif table.startswith(self.relation_prefix):
                    edges.append(table)
        return json.dumps({"nodes": nodes, "edges": edges})

    @property
    @override
    def get_structured_schema(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Return the schema of the Graph database"""
        raise NotImplementedError

    @override
    def query(
        self,
        query: str,
        params: dict[str, Value] = {},  # pyright: ignore[reportCallInDefaultInitializer]
    ) -> list[dict[str, Value]]:
        """Query the graph."""
        res = self._query(query, params)
        if "error" in res:
            raise Exception(res["error"]["message"])  # pyright: ignore[reportAny]
        else:
            result = res["result"][0]["result"]  # pyright: ignore[reportAny]
            if isinstance(result, list):
                return cast(list[dict[str, Value]], result)
            else:
                raise ValueError(
                    f"Unexpected result type: {type(result)} with value {result}"  # pyright: ignore[reportAny]
                )

    @override
    def refresh_schema(self) -> None:
        """Refresh the graph schema information."""
        raise NotImplementedError

    def delete_nodes(self, ids: list[tuple[str, str | None]] | None = None) -> None:
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
            if (
                isinstance(info, dict)
                and "tables" in info
                and isinstance(info["tables"], dict)
            ):
                for table in info["tables"].keys():
                    _ = self.connection.delete(table)
            _ = self.connection.delete(self.table_prefix + "source")

    @override
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
                            "metadata": doc.source.metadata,  # pyright: ignore[reportUnknownMemberType]
                        },
                    },
                )
                source = source["result"][0]["result"][0]  # pyright: ignore[reportAny]

            for node in doc.nodes:
                _ = self._query(
                    CREATE_NODE_QUERY,
                    {
                        "record_id": self._build_node_recordid(node),
                        "content": node.properties,  # pyright: ignore[reportUnknownMemberType]
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
                        "content": rel.properties,  # pyright: ignore[reportUnknownMemberType]
                    },
                )
