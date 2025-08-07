import enum
import logging
import time
from datetime import datetime

import click
from langchain_community.chat_loaders import WhatsAppChatLoader
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from nanoid import generate

from graphrag_travel_group_chat.chat_loaders.instagram import InstagramChatLoader
from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

from .definitions import Chunk
from .llm import infer_keywords
from .utils import format_time, get_message_timestamp_and_sender, normalize_content

# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class ChatProvider(enum.Enum):
    WHATSAPP = enum.auto()
    INSTAGRAM = enum.auto()


def ingest(
    vector_store: SurrealDBVectorStore,
    vector_store_keywords: SurrealDBVectorStore,
    graph_store: SurrealDBGraph,
    file_path: str,
    provider: ChatProvider,
    max_gap_in_s: int = 60 * 60 * 3,  # 3 hours
) -> None:
    # -- Load messages
    match provider:
        case ChatProvider.WHATSAPP:
            loader = WhatsAppChatLoader(path=file_path)
        case ChatProvider.INSTAGRAM:
            loader = InstagramChatLoader(path=file_path)
        case _:
            raise ValueError(f"Unsupported provider: {provider}")
    raw_messages = loader.lazy_load()

    # -- Create chunks based on time gaps
    chunks: list[Chunk] = []
    curr_chunk: list[str] = []
    chunk_senders = set()
    last_time = None
    for cs in raw_messages:
        for message in cs.get("messages", []):
            timestamp, sender = get_message_timestamp_and_sender(message)
            if last_time is None:
                last_time = timestamp
            if (timestamp - last_time).total_seconds() > max_gap_in_s:
                chunks.append(
                    Chunk(
                        senders=chunk_senders,
                        content="\n".join(curr_chunk),
                        timestamp=timestamp,
                    )
                )
                curr_chunk = []
                chunk_senders = set()
            curr_chunk.append(
                "\n".join(
                    [
                        f"[{timestamp}] "
                        + sender
                        + ": "
                        + normalize_content(message.content)
                    ]
                )
            )
            chunk_senders.add(sender)
            last_time = timestamp
    chunks.append(
        Chunk(
            senders=chunk_senders,
            content="\n".join(curr_chunk),
            timestamp=last_time if last_time else datetime.now(),
        )
    )
    click.echo(f"Created {len(chunks)} chunks")

    # -- Infer keywords
    keywords: set[str] = set()
    ids: list[str] = []
    documents: list[Document] = []
    for idx, chunk in enumerate(chunks):
        click.echo(f"Inferring keywords for chunk {idx}...")

        _keywords = infer_keywords(chunk.content, None)
        keywords.update(_keywords)

        id = generate()
        ids.append(id)
        documents.append(
            Document(
                page_content=chunk.content,
                metadata={
                    "message_time": format_time(chunk.timestamp),
                    "senders": list(chunk.senders),
                    "keywords": list(_keywords),
                },
            )
        )

    # -- Store keywords in vector store
    keywords_ids = list(keywords)
    click.echo(f"Adding keywords to vector store {keywords_ids}...")
    keywords_as_docs = [Document(page_content=k) for k in keywords_ids]
    vector_store_keywords.add_documents(keywords_as_docs, keywords_ids)

    # -- Store documents in vector store
    click.echo("Adding documents to vector store...")
    vector_store.add_documents(documents, ids)

    # -- Generate graph
    start_time = time.monotonic()
    click.secho("Generating graph...", fg="magenta")
    graph_documents = []
    for idx, doc in enumerate(documents):
        keyword_nodes = {
            key: Node(id=key, type="keyword", properties={"name": key})
            for key in doc.metadata.get("keywords", {})
        }
        message_node = Node(
            id=ids[idx],
            type="document",
            properties=doc.metadata | {"content": doc.page_content},
        )
        for x in doc.metadata.get("keywords", {}):
            keyword_nodes[x] = Node(id=x, type="keyword", properties={"name": x})
        nodes = [message_node] + list(keyword_nodes.values())
        relationships = [
            Relationship(
                source=message_node, target=keyword_nodes[x], type="described_by"
            )
            for x in doc.metadata.get("keywords", {})
        ]
        graph_documents.append(
            GraphDocument(nodes=nodes, relationships=relationships, source=doc)
        )

    # -- Store graph
    graph_store.add_graph_documents(graph_documents, include_source=False)
    end_time = time.monotonic()
    time_taken = end_time - start_time
    click.secho(f"\nGraph generated in {time_taken:.2f}s")
