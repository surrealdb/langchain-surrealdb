import click

from .chat import chat as chat_handler
from .db import init_stores
from .ingest import ingest as ingest_handler

ns = "langchain"
db = "example-travel-group-chat"


@click.group()
def cli() -> None: ...


@cli.command()
def ingest() -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db, clear=True)
    ingest_handler(vector_store, graph_store)
    conn.close()


@cli.command()
def chat() -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db, clear=False)
    chat_handler(conn, vector_store, graph_store, verbose=True)
    conn.close()


if __name__ == "__main__":
    cli()
