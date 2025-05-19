from typing import Generator
import asyncio

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests
from surrealdb import Surreal, AsyncSurreal

from langchain_surrealdb.vectorstores import SurrealDBVectorStore


class TestSurrealDBVectorStore(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        return False

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        conn = Surreal("ws://localhost:8000/rpc")
        conn.signin({"username": "root", "password": "root"})
        conn.use("langchain", "test")
        store = SurrealDBVectorStore(self.get_embeddings(), conn)
        store.delete()
        try:
            yield store
        finally:
            store.delete()


class TestSurrealDBVectorStoreAsync(VectorStoreIntegrationTests):
    @pytest.fixture()
    async def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""

        # Sync connection is required
        conn = Surreal("ws://localhost:8000/rpc")
        conn.signin({"username": "root", "password": "root"})
        conn.use("langchain", "test")

        async_conn = AsyncSurreal("ws://localhost:8000/rpc")
        await async_conn.signin({"username": "root", "password": "root"})
        await async_conn.use("langchain", "test")
        store = SurrealDBVectorStore(
            self.get_embeddings(), conn, async_connection=async_conn
        )
        await store.adelete()
        try:
            yield store
        finally:
            await store.adelete()
