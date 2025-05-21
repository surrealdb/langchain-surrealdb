# langchain-surrealdb

This package contains the LangChain integration with SurrealDB

>[SurrealDB](https://surrealdb.com/) is an end-to-end cloud-native database designed for modern applications, including web, mobile, serverless, Jamstack, backend, and traditional applications. With SurrealDB, you can simplify your database and API infrastructure, reduce development time, and build secure, performant apps quickly and cost-effectively.
>
>**Key features of SurrealDB include:**
>
>* **Reduces development time:** SurrealDB simplifies your database and API stack by removing the need for most server-side components, allowing you to build secure, performant apps faster and cheaper.
>* **Real-time collaborative API backend service:** SurrealDB functions as both a database and an API backend service, enabling real-time collaboration.
>* **Support for multiple querying languages:** SurrealDB supports SQL querying from client devices, GraphQL, ACID transactions, WebSocket connections, structured and unstructured data, graph querying, full-text indexing, and geospatial querying.
>* **Granular access control:** SurrealDB provides row-level permissions-based access control, giving you the ability to manage data access with precision.
>
>View the [features](https://surrealdb.com/features), the latest [releases](https://surrealdb.com/releases), and [documentation](https://surrealdb.com/docs).

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

You can run SurrealDB locally or start with a [free SurrealDB cloud account](https://surrealdb.com/docs/cloud/getting-started).

For local, two options:
1. [Install SurrealDB](https://surrealdb.com/docs/surrealdb/installation) and [run SurrealDB](https://surrealdb.com/docs/surrealdb/installation/running). Run in-memory with:

    ```bash
    surreal start -u root -p root
    ```

2. [Run with Docker](https://surrealdb.com/docs/surrealdb/installation/running/docker).

    ```bash
    docker run --rm --pull always -p 8000:8000 surrealdb/surrealdb:latest start
    ```

## Simple example

<video width="630" height="300" src="https://github.com/surrealdb/langchain-surrealdb/raw/refs/heads/docs/demo/tape/demo.webm"></video>

```python
from langchain_core.documents import Document
from langchain_surrealdb.vectorstores import SurrealDBVectorStore
from langchain_ollama import OllamaEmbeddings
from surrealdb import Surreal

conn = Surreal("ws://localhost:8000/rpc")
conn.signin({"username": "root", "password": "root"})
conn.use("langchain", "demo")
vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)

doc_1 = Document(page_content="foo", metadata={"source": "https://example.com"})
doc_2 = Document(page_content="bar", metadata={"source": "https://example.com"})

vector_store.add_documents(documents=[doc_1, doc_2], ids=["1", "2"])

results = vector_store.similarity_search_with_score(
    query="thud", k=1, custom_filter={"source": "https://example.com"}
)
for doc, score in results:
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")
```

## Next steps

- try the [jupyther notebook](./docs/vectorstores.ipynb)
- [Awesome SurrealDB](https://github.com/surrealdb/awesome-surreal), A curated list of SurrealDB resources, tools, utilities, and applications
