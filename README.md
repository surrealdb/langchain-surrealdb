<p align="center">
    <img width=120 src="https://raw.githubusercontent.com/surrealdb/icons/main/surreal.svg" />
</p>

<h3 align="center">The official SurrealDB components for LangChain</h3>

<br>

<p align="center">
    <a href="https://github.com/surrealdb/langchain-surreal"><img src="https://img.shields.io/badge/status-stable-ff00bb.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://surrealdb.com/docs/integrations/frameworks/langchain"><img src="https://img.shields.io/badge/docs-view-44cc11.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://pypi.org/project/langchain-surreal/"><img src="https://img.shields.io/pypi/v/langchain-surreal?style=flat-square"></a>
    &nbsp;
    <a href="https://pypi.org/project/langchain-surreal/"><img src="https://img.shields.io/pypi/dm/langchain-surreal?style=flat-square"></a>
    &nbsp;
    <a href="https://pypi.org/project/langchain-surreal/"><img src="https://img.shields.io/pypi/pyversions/langchain-surreal?style=flat-square"></a>
</p>

<p align="center">
    <a href="https://surrealdb.com/discord"><img src="https://img.shields.io/discord/902568124350599239?label=discord&style=flat-square&color=5a66f6"></a>
    &nbsp;
    <a href="https://twitter.com/surrealdb"><img src="https://img.shields.io/badge/twitter-follow_us-1d9bf0.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://www.linkedin.com/company/surrealdb/"><img src="https://img.shields.io/badge/linkedin-connect_with_us-0a66c2.svg?style=flat-square"></a>
    &nbsp;
    <a href="https://www.youtube.com/channel/UCjf2teVEuYVvvVC-gFZNq6w"><img src="https://img.shields.io/badge/youtube-subscribe-fc1c1c.svg?style=flat-square"></a>
</p>

# langchain-surrealdb

This package contains the LangChain integration with SurrealDB

> [SurrealDB](https://surrealdb.com/) is an end-to-end cloud-native database designed for modern applications, including
> web, mobile, serverless, Jamstack, backend, and traditional applications. With SurrealDB, you can simplify your database
> and API infrastructure, reduce development time, and build secure, performant apps quickly and cost-effectively.
>
> **Key features of SurrealDB include:**
>
> - **Reduces development time:** SurrealDB simplifies your database and API stack by removing the need for most
>   server-side components, allowing you to build secure, performant apps faster and cheaper.
> - **Real-time collaborative API backend service:** SurrealDB functions as both a database and an API backend service,
>   enabling real-time collaboration.
> - **Support for multiple querying languages:** SurrealDB supports SQL querying from client devices, GraphQL, ACID
>   transactions, WebSocket connections, structured and unstructured data, graph querying, full-text indexing, and
>   geospatial querying.
> - **Granular access control:** SurrealDB provides row-level permissions-based access control, giving you the ability to
>   manage data access with precision.
>
> View the [features](https://surrealdb.com/features), the latest [releases](https://surrealdb.com/releases),
> and [documentation](https://surrealdb.com/docs).

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

- look at the [basic example](./examples/basic). Use the Dockerfile to try it out!
- look at the [graph example](./examples/graph)
- try the [jupyther notebook](./docs/vectorstores.ipynb)
- [Awesome SurrealDB](https://github.com/surrealdb/awesome-surreal), A curated list of SurrealDB resources, tools, utilities, and applications
