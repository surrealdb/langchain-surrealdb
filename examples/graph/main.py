import click
from langchain_ollama import ChatOllama

from examples.graph.ingest import ingest as ingest_handler
from examples.graph.utils import ask, get_document_names, init_stores, vector_search
from langchain_surrealdb.graph_qa.chain import SurrealDBGraphQAChain

ns = "langchain"
db = "example-graph"


@click.group()
def cli(): ...


@cli.command()
def ingest() -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db)
    ingest_handler(vector_store, graph_store)
    conn.close()


@cli.command()
@click.option("--verbose", is_flag=True)
def chat(verbose) -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db)
    chat_model = ChatOllama(model="llama3.2", temperature=0)
    try:
        while True:
            query = click.prompt(
                click.style("\n\nWhat are your symptoms?", fg="green"), type=str
            )
            if query == "exit":
                break

            # -- Find relevant docs
            docs = vector_search(query, vector_store, k=3)
            symptoms = get_document_names(docs)

            # -- Query graph
            chain = SurrealDBGraphQAChain.from_llm(
                chat_model, graph=graph_store, verbose=verbose
            )
            ask(f"what medical practices can help with {symptoms}", chain)
            ask(f"what treatments can help with {symptoms}", chain)
    except KeyboardInterrupt:
        ...
    except Exception as e:
        print(e)  # noqa: T201

    conn.close()
    print("\nBye!\n")  # noqa: T201


if __name__ == "__main__":
    cli()
