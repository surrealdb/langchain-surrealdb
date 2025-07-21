import logging

from langchain_ollama import OllamaEmbeddings
from surrealdb import (
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
    Surreal,
)

from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

logger = logging.getLogger(__name__)


def init_stores(
    url: str = "ws://localhost:8000/rpc",
    user: str = "root",
    password: str = "root",
    *,
    ns: str = "test",
    db: str = "test",
    clear: bool = False,
) -> tuple[
    SurrealDBVectorStore,
    SurrealDBGraph,
    BlockingWsSurrealConnection | BlockingHttpSurrealConnection,
]:
    conn = Surreal(url)
    conn.signin({"username": user, "password": password})
    conn.use(ns, db)
    vector_store_ = SurrealDBVectorStore(OllamaEmbeddings(model="all-minilm:22m"), conn)
    graph_store_ = SurrealDBGraph(conn)

    if clear:
        vector_store_.delete()
        graph_store_.delete_nodes()

    # create event
    res = conn.query(r"""
        REMOVE EVENT documents_on_create ON documents;
        DEFINE EVENT documents_on_create ON documents WHEN $event = 'CREATE'
        THEN {
            UPDATE $after SET message_time = <datetime>$after.metadata.message_time;
        };
        """)
    logger.debug(f"create event documents_on_create: {res}")

    return vector_store_, graph_store_, conn
