import logging
import time
from dataclasses import asdict
from datetime import datetime

import click
from langchain_community.chat_loaders import WhatsAppChatLoader
from langchain_community.chat_loaders.utils import merge_chat_runs
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from nanoid import generate

from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

from .definitions import MessageKeywords
from .llm import infer_keywords

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def normalize_content(msg_content: str | list[str | dict]) -> str:
    if isinstance(msg_content, str):
        return msg_content
    else:
        return "\n".join(str(msg_content))


def parse_time(message_time: str | None) -> str:
    if message_time is None:
        dt = datetime.now()
    else:
        try:
            dt = datetime.strptime(message_time, "%d/%m/%Y, %H:%M:%S")
        except ValueError:
            dt = datetime.now()
    if dt.tzinfo is None:
        return f"{dt.isoformat(timespec='seconds')}+00:00"
    return dt.isoformat(timespec="seconds")


def ingest(vector_store: SurrealDBVectorStore, graph_store: SurrealDBGraph) -> None:
    # loader = WhatsAppChatLoader(path="data/_chat.txt")
    loader = WhatsAppChatLoader(path="data/_chat_test.txt")

    raw_messages = loader.lazy_load()
    # Merge consecutive messages from the same sender into a single message
    chat_sessions = merge_chat_runs(raw_messages)

    keywords: set[str] = set()
    ids: list[str] = []
    documents: list[Document] = []
    messages: list[MessageKeywords] = []
    for session in chat_sessions:
        for message in session.get("messages", []):
            events = message.additional_kwargs.get("events", [])
            if not events:
                continue
            id = generate()
            ids.append(id)
            text = normalize_content(message.content)
            _keywords = infer_keywords(text)
            keywords.union(_keywords)
            logger.info(keywords)
            documents.append(
                Document(
                    page_content=text,
                    metadata=message.additional_kwargs
                    | {
                        "message_time": parse_time(events[0].get("message_time")),
                        "keywords": list(_keywords),
                    },
                )
            )
            messages.append(
                MessageKeywords(
                    id=id,
                    sender=message.additional_kwargs.get("sender", ""),
                    keywords=_keywords,
                )
            )
    vector_store.add_documents(documents, ids)

    # -- Generate graph
    start_time = time.monotonic()
    click.secho("Generating graph...", fg="magenta")
    graph_documents = []
    for idx, doc in enumerate(documents):
        message = messages[idx]
        keyword_nodes = {
            key: Node(id=key, type="keyword", properties={"name": key})
            for key in message.keywords
        }
        message_node = Node(id=message.id, type="document", properties=asdict(message))
        sender_node = Node(
            id=message.sender, type="sender", properties={"name": message.sender}
        )
        for x in message.keywords:
            keyword_nodes[x] = Node(id=x, type="keyword", properties={"name": x})
        nodes = [sender_node, message_node] + list(keyword_nodes.values())
        relationships = [
            Relationship(source=message_node, target=sender_node, type="sent_by")
        ] + [
            Relationship(
                source=message_node, target=keyword_nodes[x], type="described_by"
            )
            for x in message.keywords
        ]
        graph_documents.append(
            GraphDocument(nodes=nodes, relationships=relationships, source=doc)
        )
    # TODO: message_node is not being created
    graph_store.add_graph_documents(graph_documents, include_source=False)
    end_time = time.monotonic()
    time_taken = end_time - start_time
    click.secho(f"\nGraph generated in {time_taken:.2f}s")
