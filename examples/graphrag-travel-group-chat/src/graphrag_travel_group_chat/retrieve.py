import logging

import click
from langchain_core.documents import Document

from langchain_surrealdb.vectorstores import SurrealDBVectorStore

logger = logging.getLogger(__name__)

NEWLINE = "\n"


def vector_search(
    query: str,
    vector_store: SurrealDBVectorStore,
    *,
    k: int = 3,
    score_threshold: float = 0.3,
) -> list[Document]:
    click.echo(f'\nSearch: "{query}"')

    # -- Similarity search
    results_w_score = vector_store.similarity_search_with_score(query, k=k)
    click.echo("\nsimilarity_search_with_score")
    for doc, score in results_w_score:
        click.secho(
            f"- [{score:.0%}] {doc.page_content[:70].split(NEWLINE)[0]}",
            fg=("green" if score >= score_threshold else "red"),
        )

    if results_w_score:
        return [doc for doc, score in results_w_score[:k] if score >= score_threshold]
    else:
        raise Exception("No results found")


# def search_close_by_time(
#     doc: Document,
#     conn: BlockingHttpSurrealConnection | BlockingWsSurrealConnection,
#     table_name: str,
# ) -> list[Document]:
#     query = r"""{{
#         let $when = <datetime>$dt;
#         return select text, metadata from {table}
#         where
#             message_time > ($when - 1h) and
#             message_time < ($when + 1h)
#         limit 50
#     }}""".format(table=table_name)
#     res = conn.query(
#         query,
#         {"dt": doc.metadata.get("message_time")},
#     )
#     if not res:
#         return []
#     if not isinstance(res, list):
#         raise RuntimeError(f"Unexpected result from DB: {type(res)}")
#     return [
#         Document(page_content=x.get("text", ""), metadata=x.get("metadata"))
#         for x in res
#     ]


def get_document_messages(docs: list[Document]) -> str:
    return "\n\n".join(
        [f"{doc.metadata.get('sender')}: {doc.page_content}" for doc in docs]
    )
