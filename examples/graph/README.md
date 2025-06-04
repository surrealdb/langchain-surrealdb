# Minimal GraphRAG

Hi!

This is an example of vector and graph search featuring:

- SurrealDBVectorStore: for similarity/relevance search
- SurrealDBGraph: a langchain graph store
- SurrealDBGraphQAChain: a Question/Answering langchain chain class capable of querying the graph using LLM models

## Running

**Requirements:**
- SurrealDB
- Ollama

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

Then, using [just](https://just.systems/man/en/packages.html) from the root directory of this repository:

```shell
just install
just examples-graph ingest
just examples-graph chat
```

Or without just:

```shell
# install deps
cd examples/graph
poetry update

# run ingest script
poetry run run ingest

# run chat script
poetry run run chat
```

## TODO
- demo the LLMGraphTransformer
- implement "few shot prompting": https://python.langchain.com/docs/tutorials/graph/#few-shot-prompting