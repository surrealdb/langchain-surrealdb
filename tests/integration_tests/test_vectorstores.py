import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from surrealdb import AsyncSurreal, Surreal
from typing_extensions import override

from langchain_surrealdb.vectorstores import (
    SurrealAsyncConnection,
    SurrealDBVectorStore,
)


class TestSurrealDBVectorStore(VectorStoreIntegrationTests):
    @property
    @override
    def has_async(self) -> bool:
        return False

    @pytest.fixture()
    @override
    def vectorstore(self) -> VectorStore:
        """Get the `VectorStore` class to test.

        The returned `VectorStore` should be empty.
        """
        conn = Surreal("ws://localhost:8000/rpc")
        _ = conn.signin({"username": "root", "password": "root"})
        conn.use("langchain", "test")
        store = SurrealDBVectorStore(self.get_embeddings(), conn)
        store.delete()
        return store


class TestSurrealDBVectorStoreAsync(VectorStoreIntegrationTests):
    @pytest.fixture()
    @override
    def vectorstore(self) -> VectorStore:
        """Get an empty vectorstore for unit tests."""

        # Sync connection is required
        conn = Surreal("ws://localhost:8000/rpc")
        _ = conn.signin({"username": "root", "password": "root"})
        conn.use("langchain", "test")

        async_conn = AsyncSurreal("ws://localhost:8000/rpc")

        async def init_async_connection(connection: SurrealAsyncConnection) -> None:
            _ = await connection.signin({"username": "root", "password": "root"})
            _ = await connection.use("langchain", "test")

        store = SurrealDBVectorStore(
            self.get_embeddings(),
            conn,
            async_connection=async_conn,
            async_initializer=init_async_connection,
        )
        store.delete()
        return store
