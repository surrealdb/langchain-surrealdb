from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from surrealdb import Surreal

from examples.graph.ingest import ingest
from langchain_surrealdb.graph_qa.chain import SurrealDBGraphQAChain
from langchain_surrealdb.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

model = "llama3.2"
chat_model = ChatOllama(model=model, temperature=0)


def init_stores() -> tuple[SurrealDBVectorStore, SurrealDBGraph]:
    conn = Surreal("ws://localhost:8000/rpc")
    conn.signin({"username": "root", "password": "root"})
    conn.use("langchain", "example-graph")
    vector_store_ = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)
    vector_store_.delete()
    graph_store_ = SurrealDBGraph(conn)
    graph_store_.delete_nodes()
    return vector_store_, graph_store_


def vector_search(query: str) -> Document:
    # -- Vector search
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
    return results_w_score[0][0]


vector_store, graph_store = init_stores()
ingest(vector_store, graph_store)

# -- User query
# query = "i have a headache"
# query = "i have an allergy"
# query = "i feel anxious"
query = "a have a runny nose and my eyes feel itchy"
doc = vector_search(query)

# -- Query Graph
chain = SurrealDBGraphQAChain.from_llm(chat_model, graph=graph_store, verbose=False)


def ask(q: str, chain: SurrealDBGraphQAChain) -> None:
    print(f"\nQuestion: {q}\n=========\n")
    _response = chain.invoke({"query": q})
    print("\nAnswer:\n=======\n", _response["result"][0]["text"])  # noqa: T201


ask(f"what medical practices can help with {doc.metadata['name']}", chain)
ask(f"what treatments can help with {doc.metadata['name']}", chain)
