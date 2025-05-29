import click
from langchain_ollama import ChatOllama

from langchain_surrealdb.experimental.graph_qa.chain import SurrealDBGraphQAChain

from .ingest import ingest as ingest_handler
from .utils import ask, get_document_names, init_stores, vector_search

ns = "langchain"
db = "example-graph"


@click.group()
def cli() -> None: ...


@cli.command()
def ingest() -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db)
    ingest_handler(vector_store, graph_store)
    conn.close()


@cli.command()
@click.option("--verbose", is_flag=True)
def chat(verbose: bool) -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db)
    chat_model = ChatOllama(model="llama3.2", temperature=0)

    def query_logger(q: str, results: int) -> None:
        conn.insert("generated_query", {"query": q, "results": results})

    try:
        while True:
            query = click.prompt(
                click.style("\nWhat are your symptoms?", fg="green"), type=str
            )
            if query == "exit":
                break

            # -- Find relevant docs
            docs = vector_search(query, vector_store, k=3)
            symptoms = get_document_names(docs)

            # -- Query graph
            chain = SurrealDBGraphQAChain.from_llm(
                chat_model,
                graph=graph_store,
                verbose=verbose,
                query_logger=query_logger,
            )
            ask(f"what medical practices can help with {symptoms}", chain)
            ask(f"what treatments can help with {symptoms}", chain)
    except KeyboardInterrupt:
        ...
    except Exception as e:
        print(e)  # noqa: T201

    conn.close()
    print("Bye!")  # noqa: T201


if __name__ == "__main__":
    cli()
