import logging

import click
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
from .retrieve import get_document_messages, vector_search

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
            docs = vector_search(query, vector_store, k=5, score_threshold=0.3)
            messages = get_document_messages(docs)

            # chat_model = ChatOllama(model="llama3.2", temperature=0)
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
            _keywords: list[str] = []
            _inferred = infer_keywords(query)
            click.secho(f"Query keywords: {_inferred}", fg="green")
            chain = SurrealDBGraphQAChain.from_llm(
                chat_model,
                graph=graph_store,
                verbose=verbose,
                query_logger=query_logger,
            )
            _ask(
                f"""
Execute a query like this:

SELECT <-relation_described_by<-graph_document.content as doc
    FROM graph_keyword WHERE name IN ["watch", "movie"]

But using these keywords: {list(_inferred)}

With the result, answer this query: "{query}"
""",
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
