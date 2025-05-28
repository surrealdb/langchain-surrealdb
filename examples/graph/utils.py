import click
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from surrealdb import Surreal

from langchain_surrealdb.graph_qa.chain import SurrealDBGraphQAChain
from langchain_surrealdb.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore


def init_stores(clear: bool = False) -> tuple[SurrealDBVectorStore, SurrealDBGraph]:
    conn = Surreal("ws://localhost:8000/rpc")
    conn.signin({"username": "root", "password": "root"})
    conn.use("langchain", "example-graph")
    vector_store_ = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)
    graph_store_ = SurrealDBGraph(conn)
    if clear:
        vector_store_.delete()
        graph_store_.delete_nodes()
    return vector_store_, graph_store_


def vector_search(query: str, vector_store: SurrealDBVectorStore) -> Document:
    print(f'\nSearch: "{query}"')  # noqa: T201
    results = vector_store.max_marginal_relevance_search(
        query, k=3, fetch_k=20, score_threshold=0.3
    )
    print("\nmax_marginal_relevance_search:")  # noqa: T201
    for doc in results:
        print(f"- {doc.page_content}")  # noqa: T201
    results_w_score = vector_store.similarity_search_with_score(query, k=3)
    print("\nsimilarity_search_with_score")  # noqa: T201
    for doc, score in results_w_score:
        print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201
    if results_w_score:
        return results_w_score[0][0]
    else:
        raise Exception("No results found")


def ask(q: str, chain: SurrealDBGraphQAChain) -> None:
    print(f"\nQuestion: {q}")
    _response = chain.invoke({"query": q})
    print("Answer:\n=======\n", _response["result"][0]["text"])  # noqa: T201
