import logging

import click
from langchain_ollama import ChatOllama
from surrealdb import (
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
)

from langchain_surrealdb.experimental.graph_qa.chain import SurrealDBGraphQAChain
from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

from .llm import generate_answer_from_messages, infer_keywords, summarize_answer
from .retrieve import format_document_messages, vector_search

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def chat(
    conn: BlockingWsSurrealConnection | BlockingHttpSurrealConnection,
    vector_store: SurrealDBVectorStore,
    graph_store: SurrealDBGraph,
    *,
    verbose: bool,
) -> None:
    chat_model = ChatOllama(model="llama3.2", temperature=0)

    def query_logger(q: str, results: int) -> None:
        conn.insert("generated_query", {"query": q, "results": results})

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
            docs = vector_search(query, vector_store, k=5, score_threshold=0.3)
            messages = format_document_messages(docs)

            similarity_answer = generate_answer_from_messages(
                messages, query, user_name
            )
            click.secho(similarity_answer, fg="blue")

            # -- Query graph
            _inferred = infer_keywords(query, all_keywords)
            click.secho(f"Query keywords: {_inferred}", fg="green")
            chain = SurrealDBGraphQAChain.from_llm(
                chat_model,
                graph=graph_store,
                verbose=verbose,
                query_logger=query_logger,
            )
            graph_answer = _ask(
                f"""
Execute a query like this:

SELECT <-relation_described_by<-graph_document.content as doc
    FROM graph_keyword WHERE name IN ["watch", "movie"]

But using these keywords: {list(_inferred)}

With the result, answer this query: "{query}"
""",
                chain,
            )

            # -- Summarize final answer
            final_answer = summarize_answer(
                [similarity_answer, graph_answer], query, user_name
            )
            click.echo("\n\n")
            click.echo("======================================================")
            click.secho(final_answer, fg="magenta")
            click.echo("======================================================")
    except KeyboardInterrupt:
        ...
    except Exception as e:
        click.echo(e)

    conn.close()
    click.echo("Bye!")


def _ask(q: str, chain: SurrealDBGraphQAChain) -> str:
    print(click.style("Loading...", fg="magenta"), end="", flush=True)  # noqa: T201
    response = chain.invoke({"query": q})
    print(click.style("\rAnswer: ", fg="blue"), end="")  # noqa: T201
    click.echo(response["result"][0]["text"])
    return response["result"][0]["text"]
