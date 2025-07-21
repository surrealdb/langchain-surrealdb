import click

from .chat import chat as chat_handler
from .db import init_stores
from .ingest import ChatProvider
from .ingest import ingest as ingest_handler

ns = "langchain"
db = "example-travel-group-chat"


@click.group()
def cli() -> None: ...


@cli.command()
@click.argument("provider", type=click.Choice(ChatProvider, case_sensitive=False))
@click.argument("file", type=click.Path(exists=True))
def ingest(file: str, provider: ChatProvider) -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db, clear=True)
    ingest_handler(vector_store, graph_store, file, provider, max_gap_in_s=60 * 60 * 3)
    conn.close()


@cli.command()
def chat() -> None:
    vector_store, graph_store, conn = init_stores(ns=ns, db=db, clear=False)
    chat_handler(conn, vector_store, graph_store, verbose=True)
    conn.close()


if __name__ == "__main__":
    cli()
