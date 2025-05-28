import click
from langchain_ollama import ChatOllama

from examples.graph.ingest import ingest as ingest_handler
from examples.graph.utils import init_stores, vector_search, ask
from langchain_surrealdb.graph_qa.chain import SurrealDBGraphQAChain


@click.group()
def cli():
    pass


@cli.command()
def ingest() -> None:
    vector_store, graph_store = init_stores()
    ingest_handler(vector_store, graph_store)


@cli.command()
def chat() -> None:
    vector_store, graph_store = init_stores()
    model = "llama3.2"
    chat_model = ChatOllama(model=model, temperature=0)

    while True:
        query = click.prompt("\n\nWhat are your symptoms?", type=str)
        if query == "/exit":
            break
        try:
            doc = vector_search(query, vector_store)

            # -- Graph chain
            chain = SurrealDBGraphQAChain.from_llm(
                chat_model, graph=graph_store, verbose=False
            )

            # -- Query graph
            ask(f"what medical practices can help with {doc.metadata['name']}", chain)
            ask(f"what treatments can help with {doc.metadata['name']}", chain)
        except Exception as e:
            print(e)  # noqa: T201

    print("\nBye!\n")


if __name__ == "__main__":
    cli()
