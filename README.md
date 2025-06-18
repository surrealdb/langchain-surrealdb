<p align="center">
    <img width=120 src="https://raw.githubusercontent.com/surrealdb/icons/main/surreal.svg" />
</p>

<h3 align="center">The official SurrealDB components for LangChain</h3>

<br>

<p align="center">
    <a href="https://github.com/surrealdb/langchain-surrealdb"><img src="https://img.shields.io/badge/status-stable-ff00bb.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://surrealdb.com/docs/integrations/frameworks/langchain"><img src="https://img.shields.io/badge/docs-view-44cc11.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://pypi.org/project/langchain-surrealdb/"><img src="https://img.shields.io/pypi/v/langchain-surrealdb?style=flat-square"></a>
    &nbsp;
    <a href="https://pypi.org/project/langchain-surrealdb/"><img src="https://img.shields.io/pypi/dm/langchain-surrealdb?style=flat-square"></a>
    &nbsp;
    <a href="https://pypi.org/project/langchain-surrealdb/"><img src="https://img.shields.io/pypi/pyversions/langchain-surrealdb?style=flat-square"></a>
</p>

<p align="center">
    <a href="https://surrealdb.com/discord"><img src="https://img.shields.io/discord/902568124350599239?label=discord&style=flat-square&color=5a66f6"></a>
    &nbsp;
    <a href="https://x.com/surrealdb"><img alt="X (formerly Twitter) Follow" src="https://img.shields.io/twitter/follow/surrealdb?style=flat-square&logo=x&label=follow%20us"></a>
    &nbsp;
    <a href="https://www.linkedin.com/company/surrealdb/"><img src="https://img.shields.io/badge/linkedin-connect_with_us-0a66c2.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://www.youtube.com/channel/UCjf2teVEuYVvvVC-gFZNq6w"><img src="https://img.shields.io/badge/youtube-subscribe-fc1c1c.svg?style=flat-square"></a>
</p>

# langchain-surrealdb

This package contains the LangChain integration with SurrealDB

> [SurrealDB](https://surrealdb.com/) is a unified, multi-model database purpose-built for AI systems. It combines structured and unstructured data (including vector search, graph traversal, relational queries, full-text search, document storage, and time-series data) into a single ACID-compliant engine, scaling from a 3 MB edge binary to petabyte-scale clusters in the cloud. By eliminating the need for multiple specialized stores, SurrealDB simplifies architectures, reduces latency, and ensures consistency for AI workloads.
>
> **Why SurrealDB Matters for GenAI Systems**
> - **One engine for storage and memory:** Combine durable storage and fast, agent-friendly memory in a single system, providing all the data your agent needs and removing the need to sync multiple systems.
> - **One-hop memory for agents:** Run vector search, graph traversal, semantic joins, and transactional writes in a single query, giving LLM agents fast, consistent memory access without stitching relational, graph and vector databases together.
> - **In-place inference and real-time updates:** SurrealDB enables agents to run inference next to data and receive millisecond-fresh updates, critical for real-time reasoning and collaboration.
> - **Versioned, durable context:** SurrealDB supports time-travel queries and versioned records, letting agents audit or “replay” past states for consistent, explainable reasoning.
> - **Plug-and-play agent memory:** Expose AI memory as a native concept, making it easy to use SurrealDB as a drop-in backend for AI frameworks.

## Installation

```bash
# -- Using pip
pip install -U langchain-surrealdb surrealdb
# -- Using poetry
poetry add langchain-surrealdb surrealdb
# -- Using uv
uv add --upgrade langchain-surrealdb surrealdb
```

## Requirements

You can run SurrealDB locally or start with
a [free SurrealDB cloud account](https://surrealdb.com/docs/cloud/getting-started).

For local, two options:

1. [Install SurrealDB](https://surrealdb.com/docs/surrealdb/installation)
  and [run SurrealDB](https://surrealdb.com/docs/surrealdb/installation/running). Run in-memory with:

  ```bash
  surreal start -u root -p root
  ```

2. [Run with Docker](https://surrealdb.com/docs/surrealdb/installation/running/docker).

  ```bash
  docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start
  ```

## Simple example

<video width="630" height="300" src="https://github.com/user-attachments/assets/9e1c0dda-4334-48ea-8317-d2dc72b275c0"></video>

```python
from langchain_core.documents import Document
from langchain_surrealdb.vectorstores import SurrealDBVectorStore
from langchain_ollama import OllamaEmbeddings
from surrealdb import Surreal

conn = Surreal("ws://localhost:8000/rpc")
conn.signin({"username": "root", "password": "root"})
conn.use("langchain", "demo")
vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)

doc_1 = Document(page_content="foo", metadata={"source": "https://surrealdb.com"})
doc_2 = Document(page_content="SurrealDB", metadata={"source": "https://surrealdb.com"})

vector_store.add_documents(documents=[doc_1, doc_2], ids=["1", "2"])

results = vector_store.similarity_search_with_score(
    query="surreal", k=1, custom_filter={"source": "https://surrealdb.com"}
)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

## Next steps

- look at the [basic example](https://github.com/surrealdb/langchain-surrealdb/tree/main/examples/basic). Use the Dockerfile to try it out!
- look at the [graph example](https://github.com/surrealdb/langchain-surrealdb/tree/main/examples/graph)
- try the [jupyter notebook](https://github.com/surrealdb/langchain-surrealdb/tree/main/docs/vectorstores.ipynb)
- [Awesome SurrealDB](https://github.com/surrealdb/awesome-surreal), A curated list of SurrealDB resources, tools, utilities, and applications
