from textwrap import dedent

import click
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from surrealdb import (
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
)

from langchain_surrealdb.experimental.graph_qa.chain import SurrealDBGraphQAChain
from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

NEWLINE = "\n"


def vector_search(
    query: str,
    vector_store: SurrealDBVectorStore,
    *,
    k: int = 3,
    score_threshold: float = 0.3,
    verbose: bool = True,
) -> list[tuple[Document, float]]:
    results_w_score = vector_store.similarity_search_with_score(query, k=k)
    if verbose:
        click.echo("\nsimilarity_search_with_score:")
        for doc, score in results_w_score:
            click.secho(
                f"- [{score:.0%}] {doc.page_content[:70].split(NEWLINE)[0]}",
                fg=("green" if score >= score_threshold else "red"),
            )

    if results_w_score:
        return [
            (doc, score)
            for doc, score in results_w_score[:k]
            if score >= score_threshold
        ]
    else:
        raise Exception("No results found")


def format_document_messages(docs: list[Document]) -> list[str]:
    result = [doc.page_content for doc in docs]
    return result


def graph_qa(
    graph_store: SurrealDBGraph,
    similar_keyword_docs: list[tuple[Document, float]],
    query: str,
    verbose: bool,
) -> str:
    def query_logger(q: str, results: int) -> None:
        graph_store.connection.insert(
            "generated_query", {"query": q, "results": results}
        )

    chat_model = ChatOllama(model="llama3.2", temperature=0)
    chain = SurrealDBGraphQAChain.from_llm(
        chat_model,
        graph=graph_store,
        verbose=verbose,
        query_logger=query_logger,
    )
    similar_keywords = [x.page_content for x, _score in similar_keyword_docs]
    response = chain.invoke(
        {
            "query": dedent(f"""
            Execute a query like this:

            SELECT <-relation_described_by<-graph_document.content as doc
                FROM graph_keyword WHERE name IN ["watch", "movie"]

            But using these keywords: {list(similar_keywords)}

            With the result, answer this query: "{query}"
            """)
        }
    )
    graph_answer = response["result"][0]["text"]
    return graph_answer


def graph_query(
    conn: BlockingWsSurrealConnection | BlockingHttpSurrealConnection,
    similar_keyword_docs: list[tuple[Document, float]],
) -> list[str]:
    similar_keywords = [x.page_content for x, _score in similar_keyword_docs]
    query = dedent("""
        SELECT id, content from array::flatten(
            SELECT VALUE doc FROM (
                SELECT (<-relation_described_by<-graph_document.{content,id}) AS doc
                FROM graph_keyword WHERE name IN $kws
            )
        )
        GROUP BY id
    """)
    result = conn.query(query, {"kws": similar_keywords})
    if isinstance(result, list):
        result = [x.get("content", []) for x in result]
    else:
        result = []
    return result
