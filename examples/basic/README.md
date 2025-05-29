# Basic vector search example

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
just examples-basic ru
```

