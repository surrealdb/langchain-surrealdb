import logging

import click
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from surrealdb import (
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
)

from langchain_surrealdb.experimental.graph_qa.chain import SurrealDBGraphQAChain
from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

from .llm import infer_keywords
from .retrieve import get_document_messages, search_close_by_time, vector_search

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

    try:
        while True:
            query = click.prompt(
                click.style("\nAsk something about the group chat", fg="green"),
                type=str,
            )
            if query == "exit":
                break

            # -- Find relevant docs
            docs = vector_search(query, vector_store, k=5)

            # -- Get recent messages (before and after)
            docs_window: list[Document] = []
            for doc in docs:
                docs_window += search_close_by_time(doc, conn, vector_store.table)

            click.secho(f"Retrieved {len(docs_window)}", fg="yellow")
            messages = get_document_messages(docs_window)

            chat_model = ChatOllama(model="llama3.2", temperature=0)
            prompt = ChatPromptTemplate(
                [
                    ("system", "Role: You are a very helpful assitant"),
                    (
                        "system",
                        "Goal: Answer questions based on messages from a group chat",
                    ),
                    ("system", f"--Messages--:\n\n{messages}"),
                    ("user", query),
                ]
            )
            chain = prompt | chat_model
            res = chain.invoke({"messages": messages})
            click.secho(res.content, fg="blue")

            # -- Query graph
            _keywords: list[str] = (
                docs[0].metadata.get("keywords", [])
                + docs[1].metadata.get("keywords", [])
                + docs[2].metadata.get("keywords", [])
            )
            query_keywords = list(infer_keywords(query) | set(_keywords))
            chain = SurrealDBGraphQAChain.from_llm(
                chat_model,
                graph=graph_store,
                verbose=verbose,
                query_logger=query_logger,
            )
            _ask(
                f"""
Find the answer to: "{query}" using the following keywords:

{query_keywords}""",
                chain,
            )
    except KeyboardInterrupt:
        ...
    except Exception as e:
        click.echo(e)

    conn.close()
    click.echo("Bye!")


def _ask(q: str, chain: SurrealDBGraphQAChain) -> None:
    # print(click.style("\nQuestion: ", fg="blue"), end="")  # noqa: T201
    # print(q)  # noqa: T201
    print(click.style("Loading...", fg="magenta"), end="", flush=True)  # noqa: T201
    response = chain.invoke({"query": q})
    print(click.style("\rAnswer: ", fg="blue"), end="")  # noqa: T201
    print(response["result"][0]["text"])  # noqa: T201
