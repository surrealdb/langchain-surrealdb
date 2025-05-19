# langchain-surrealdb

This package contains the LangChain integration with SurrealDB

## Installation

```bash
pip install -U langchain-surrealdb
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatSurrealDB` class exposes chat models from SurrealDB.

```python
from langchain_surrealdb import ChatSurrealDB

llm = ChatSurrealDB()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`SurrealDBEmbeddings` class exposes embeddings from SurrealDB.

```python
from langchain_surrealdb import SurrealDBEmbeddings

embeddings = SurrealDBEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`SurrealDBLLM` class exposes LLMs from SurrealDB.

```python
from langchain_surrealdb import SurrealDBLLM

llm = SurrealDBLLM()
llm.invoke("The meaning of life is")
```
