import click
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from surrealdb import (
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
    Surreal,
)

from langchain_surrealdb.graph_qa.chain import SurrealDBGraphQAChain
from langchain_surrealdb.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore


def init_stores(
    url="ws://localhost:8000/rpc",
    user="root",
    password="root",
    *,
    ns="test",
    db="test",
    clear: bool = False,
) -> tuple[
    SurrealDBVectorStore,
    SurrealDBGraph,
    BlockingWsSurrealConnection | BlockingHttpSurrealConnection,
]:
    conn = Surreal(url)
    conn.signin({"username": user, "password": password})
    conn.use(ns, db)
    vector_store_ = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)
    graph_store_ = SurrealDBGraph(conn)
    if clear:
        vector_store_.delete()
        graph_store_.delete_nodes()
    return vector_store_, graph_store_, conn


def vector_search(
    query: str, vector_store: SurrealDBVectorStore, *, k=3
) -> list[Document]:
    print(f'\nSearch: "{query}"')  # noqa: T201

    # -- Max marginal relevance search
    results = vector_store.max_marginal_relevance_search(
        query, k=k, fetch_k=20, score_threshold=0.3
    )
    print("\nmax_marginal_relevance_search:")  # noqa: T201
    for doc in results:
        print(f"- {doc.page_content}")  # noqa: T201

    # -- Similarity search
    results_w_score = vector_store.similarity_search_with_score(query, k=k)
    print("\nsimilarity_search_with_score")  # noqa: T201
    for doc, score in results_w_score:
        print(f"- [{score:.0%}] {doc.page_content}")  # noqa: T201

    if results_w_score:
        return [doc for doc, _ in results_w_score[:k]]
    else:
        raise Exception("No results found")


def ask(q: str, chain: SurrealDBGraphQAChain) -> None:
    print(click.style("\nQuestion: ", fg="blue"), end="")  # noqa: T201
    print(q)  # noqa: T201
    print(click.style("Loading...", fg="magenta"), end="", flush=True)  # noqa: T201
    response = chain.invoke({"query": q})
    print(click.style("\rAnswer: ", fg="blue"), end="")  # noqa: T201
    print(response["result"][0]["text"])  # noqa: T201


def get_document_names(docs: list[Document]) -> str:
    return ", ".join([doc.metadata["name"] for doc in docs])
