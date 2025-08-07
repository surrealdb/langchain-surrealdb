import time

import click
from langchain_core.documents import Document
from surrealdb import (
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
)

from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

from .llm import generate_answer_from_messages, infer_keywords, summarize_answer
from .retrieve import format_document_messages, graph_query, vector_search


def chat(
    conn: BlockingWsSurrealConnection | BlockingHttpSurrealConnection,
    vector_store: SurrealDBVectorStore,
    vector_store_keywords: SurrealDBVectorStore,
    graph_store: SurrealDBGraph,
    *,
    verbose: bool,
) -> None:
    res = conn.query("SELECT VALUE name FROM graph_keyword")
    assert isinstance(res, list)
    all_keywords: list[str] = [k for k in res if isinstance(k, str)]

    user_name = click.prompt(
        click.style("What's your name", fg="green"),
        type=str,
    )

    try:
        while True:
            query = click.prompt(
                click.style("\nAsk something about the group chat", fg="green"),
                type=str,
            )
            if query == "exit":
                break

            # -- Find relevant docs
            click.echo("\n======================================================")
            start_time_similarity = time.time()
            docs = vector_search(query, vector_store, k=5, score_threshold=0.3)
            messages = format_document_messages([doc for doc, _score in docs])

            similarity_answer = generate_answer_from_messages(
                messages, query, user_name
            )
            end_time_similarity = time.time()
            click.secho("\nAnswer from vector search: ", fg="blue")
            click.secho(similarity_answer, fg="magenta")
            click.secho(
                f"took: {end_time_similarity - start_time_similarity}", fg="red"
            )

            # -- Search keywords
            click.echo("\n======================================================")
            start_time_graph = time.time()
            _inferred = infer_keywords(query, all_keywords)
            click.secho(f"Query keywords: {_inferred}")
            similar_keyword_docs: list[tuple[Document, float]] = []
            for k in _inferred:
                tmp = vector_search(
                    k, vector_store_keywords, k=2, score_threshold=0.5, verbose=False
                )
                similar_keyword_docs.extend(tmp)
            similar_keyword_docs = sorted(
                similar_keyword_docs, key=lambda x: x[1], reverse=True
            )
            click.secho(f"Similar keywords found: {similar_keyword_docs}", fg="green")

            # -- Query graph

            # graph_qa uses SurrealDBGraphQAChain, which uses the LLM to
            # generate the Surql, which is an overkill
            # graph_answer = graph_qa(graph_store, similar_keyword_docs, query, verbose)

            # graph_query uses a pre-defined query template and executes it on
            # the graph database
            graph_results = graph_query(conn, similar_keyword_docs)
            graph_answer = generate_answer_from_messages(
                graph_results, query, user_name
            )

            end_time_graph = time.time()
            click.secho("\nAnswer from graph: ", fg="blue")
            click.secho(graph_answer, fg="magenta")
            click.secho(f"took: {end_time_graph - start_time_graph}", fg="red")

            # -- Summarize final answer
            click.echo("\n======================================================")
            dedupped = list(set(messages + graph_results))
            len_all = len(messages) + len(graph_results)
            len_dedupped = len(dedupped)
            if len_dedupped != len_all:
                click.secho(f"Dedupped from {len_all} to {len_dedupped}", fg="yellow")
            final_answer = summarize_answer(
                dedupped,
                query,
                user_name,
            )
            click.secho("Summary from both:", fg="blue")
            click.secho(final_answer, fg="magenta")
            click.echo("\n======================================================")
    except KeyboardInterrupt:
        ...
    except Exception as e:
        click.echo(e)

    conn.close()
    click.echo("Bye!")
