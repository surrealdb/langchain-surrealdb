"""SurrealDB vector stores."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Callable, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import (
    maximal_marginal_relevance,  # pyright: ignore[reportUnknownVariableType]
)
from surrealdb import (
    AsyncHttpSurrealConnection,
    AsyncWsSurrealConnection,
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
    RecordID,
    Value,
)
from typing_extensions import override

from langchain_surrealdb.utils import extract_id

SurrealConnection = BlockingWsSurrealConnection | BlockingHttpSurrealConnection
SurrealAsyncConnection = AsyncWsSurrealConnection | AsyncHttpSurrealConnection
AsyncConnectionInitializer = Callable[[SurrealAsyncConnection], Awaitable[None]]
CustomFilter = dict[str, Value]
QueryArgs = dict[str, Value]

GET_BY_ID_QUERY = """
    SELECT *
    FROM type::table($table)
    WHERE id IN array::combine([$table], $ids)
        .map(|$v| type::thing($v[0], $v[1])) \
"""

DEFINE_INDEX = """
    DEFINE INDEX IF NOT EXISTS {index_name}
        ON TABLE {table}
        FIELDS vector
        MTREE DIMENSION {embedding_dimension} DIST COSINE TYPE F32
        CONCURRENTLY;
"""

SEARCH_QUERY = """
    SELECT
        id,
        text,
        metadata,
        vector,
        similarity
    FROM (
        SELECT
            id,
            text,
            metadata,
            vector,
            (1 - vector::distance::knn()) as similarity
        FROM type::table($table)
        WHERE vector <|{k}|> $vector
            {custom_filter_str}
    )
    WHERE similarity >= $score_threshold
    ORDER BY similarity DESC
"""


@dataclass
class SurrealDocument:
    _: KW_ONLY
    id: RecordID = field(hash=False)
    text: str
    vector: list[float]
    similarity: float | None = None
    metadata: dict[str, Value] = field(default_factory=dict)

    def into(self) -> Document:
        return Document(
            id=str(self.id.id),  # pyright: ignore[reportAny]
            page_content=self.text,
            metadata=self.metadata,
        )


class SurrealDBVectorStore(VectorStore):
    """SurrealDB vector store integration.

    Setup:
        Install ``langchain-surrealdb`` and ``surrealdb``.

        .. code-block:: bash

            pip install -U langchain-surrealdb surrealdb

    Key init args — indexing params:
        embedding: Embeddings
            Embedding function to use.
        table: str = "documents"
            Name for the table.
        index_name: str = "documents_vector_index"
            Name for the vector index.
        embedding_dimension: int | None = None
            Embedding vector dimension. If not provided, it will be calculated using the embedding function.

    Key init args — client params:
        connection: Union[BlockingWsSurrealConnection, BlockingHttpSurrealConnection]
            SurrealDB blocking connection.
        async_connection: SurrealAsyncConnection | None = None
            Optional async SurrealDB connection. Required only if you want to use the available async functions. All
            async functions are available as blocking functions, but not vice versa.

    Install and start SurrealDB:
        [Install SurrealDB](https://surrealdb.com/docs/surrealdb/installation).

        Then start SurrealDB:

        .. code-block:: bash

            surrealdb start -u root -p root

        This command starts SurrealDB in memory. For more options: [Running SurrealDB](https://surrealdb.com/docs/surrealdb/installation/running).

    Instantiate:
        .. code-block:: python

            from langchain_surrealdb.vectorstores import SurrealDBVectorStore
            from langchain_ollama import OllamaEmbeddings

            conn = Surreal("ws://localhost:8000/rpc")
            conn.signin({"username": "root", "password": "root"})
            conn.use("langchain", "demo")
            vector_store = SurrealDBVectorStore(
                OllamaEmbeddings(model="llama3.2"),
                conn
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(
                query="surreal", k=1, custom_filter={"source": "https://surrealdb.com"}
            )

        .. code-block:: python

            [Document(id='4', metadata={'source': 'https://surrealdb.com'}, page_content='this is surreal')]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(
                query="thud", k=1, custom_filter={"source": "https://surrealdb.com"}
            )
            for doc, score in results:
                print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201

        .. code-block:: python

            [similarity=57%] this is surreal

    Async:
        .. code-block:: python

            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            await vector_store.adelete(ids=["3"])

            # search
            results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(
                query="thud", k=1, custom_filter={"source": "https://surrealdb.com"}
            )
            for doc, score in results:
                print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201

        .. code-block:: python

            [similarity=57%] this is surreal

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 1, "lambda_mult": 0.5}
            )
            retriever.invoke("surreal")

        .. code-block:: python

            [Document(id='4', metadata={'source': 'https://surrealdb.com'}, page_content='this is surreal')]

    """  # noqa: E501

    def __init__(
        self,
        embedding: Embeddings,
        connection: SurrealConnection,
        table: str = "documents",
        index_name: str = "documents_vector_index",
        embedding_dimension: int | None = None,
        async_connection: SurrealAsyncConnection | None = None,
        async_initializer: AsyncConnectionInitializer | None = None,
    ) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        self.embedding: Embeddings = embedding
        self.table: str = table
        self.index_name: str = index_name
        self.connection: SurrealConnection = connection
        self.async_connection: SurrealAsyncConnection | None = async_connection
        self._async_initializer: AsyncConnectionInitializer | None = async_initializer
        self._async_initialized: bool = False
        self._async_initializer_lock: asyncio.Lock | None = None
        if embedding_dimension is not None:
            self.embedding_dimension: int = embedding_dimension
        else:
            self.embedding_dimension = len(self.embedding.embed_query("foo"))
        self._ensure_index()

    async def _ensure_async_connection_ready(self) -> None:
        """Run lazy async connection setup once per instance."""
        if (
            self.async_connection is None
            or self._async_initializer is None
            or self._async_initialized
        ):
            return
        if self._async_initializer_lock is None:
            self._async_initializer_lock = asyncio.Lock()
        async with self._async_initializer_lock:
            if self._async_initialized:
                return
            await self._async_initializer(self.async_connection)
            self._async_initialized = True

    def _ensure_index(self) -> None:
        query = DEFINE_INDEX.format(
            index_name=self.index_name,
            table=self.table,
            embedding_dimension=self.embedding_dimension,
        )
        _ = self.connection.query(query)

    @staticmethod
    def _parse_documents(ids: Sequence[str], results: Value) -> list[Document]:
        if not isinstance(results, list):
            raise ValueError("Invalid query results, expected a list")
        results_cast: list[dict[str, Value]] = cast(list[dict[str, Value]], results)
        docs: dict[str, Document] = {}
        for x in results_cast:
            id = x.get("id")
            if not isinstance(id, RecordID):
                raise ValueError("Invalid query results, expected a RecordID")
            text = x.get("text")
            vector = x.get("vector")
            metadata = x.get("metadata")
            similarity = x.get("similarity")
            doc = SurrealDocument(
                id=id,
                text=str(text),
                vector=cast(list[float], vector),
                metadata=cast(dict[str, Value], metadata),
                similarity=cast(float, similarity),
            ).into()
            if doc.id is not None:
                docs[doc.id] = doc
        # sort docs in the same order as the passed in IDs
        result: list[Document] = []
        for key in ids:
            d = docs[key]
            result.append(d)
        return result

    @staticmethod
    def _parse_results(results: Value) -> list[tuple[Document, float, list[float]]]:
        if not isinstance(results, list):
            raise ValueError("Invalid query results, expected a list")
        results_cast: list[dict[str, Value]] = cast(list[dict[str, Value]], results)
        parsed: list[tuple[Document, float, list[float]]] = []
        for raw in results_cast:
            id = raw.get("id")
            if not isinstance(id, RecordID):
                raise ValueError("Invalid query results, expected a RecordID")
            text = raw.get("text")
            vector: list[float] = cast(list[float], raw.get("vector"))
            metadata = cast(dict[str, Value], raw.get("metadata"))
            similarity: float = cast(float, raw.get("similarity"))
            parsed.append(
                (
                    SurrealDocument(
                        id=id,
                        text=str(text),
                        vector=vector,
                        metadata=metadata,
                        similarity=similarity,
                    ).into(),
                    similarity,
                    vector,
                )
            )
        return parsed

    @classmethod
    @override
    def from_texts(
        cls: type[SurrealDBVectorStore],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,  # pyright: ignore[reportExplicitAny]
        *,
        ids: list[str] | None = None,
        connection: SurrealConnection | None = None,
        table: str = "documents",
        index_name: str = "documents_vector_index",
        embedding_dimension: int | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> SurrealDBVectorStore:
        if connection is None:
            raise ValueError("Connection is required")
        store = cls(
            embedding=embedding,
            connection=connection,
            table=table,
            index_name=index_name,
            embedding_dimension=embedding_dimension,
        )
        _ = store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)  # pyright: ignore[reportUnknownMemberType]
        return store

    @classmethod
    @override
    async def afrom_texts(
        cls: type[SurrealDBVectorStore],
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,  # pyright: ignore[reportExplicitAny]
        *,
        ids: list[str] | None = None,
        connection: SurrealConnection | None = None,
        async_connection: SurrealAsyncConnection | None = None,
        table: str = "documents",
        index_name: str = "documents_vector_index",
        embedding_dimension: int | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> SurrealDBVectorStore:
        if connection is None:
            raise ValueError("Connection is required")
        store = cls(
            embedding=embedding,
            connection=connection,
            async_connection=async_connection,
            table=table,
            index_name=index_name,
            embedding_dimension=embedding_dimension,
        )
        _ = await store.aadd_texts(texts=texts, metadatas=metadatas, **kwargs)  # pyright: ignore[reportUnknownMemberType, reportAny]
        return store

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _prepare_documents(
        self, documents: list[Document], ids: list[str] | None
    ) -> tuple[list[list[float]], Iterator[str | None]]:
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[str | None] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        return vectors, id_iterator

    @override
    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[str]:
        """Add documents to the store."""
        vectors, id_iterator = self._prepare_documents(documents, ids)
        ids_: list[str] = []
        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_data: dict[str, Value] = {
                "vector": cast(list[Value], vector),
                "text": doc.page_content,
                "metadata": doc.metadata,  # pyright: ignore[reportUnknownMemberType]
            }
            if doc_id is not None:
                record_id = RecordID(self.table, doc_id)
                inserted = self.connection.upsert(record_id, doc_data)
            else:
                inserted = self.connection.insert(self.table, doc_data)
            if isinstance(inserted, list):
                for record in inserted:
                    ids_.append(extract_id(record))
            elif isinstance(inserted, dict):
                ids_.append(extract_id(inserted))
        return ids_

    @override
    async def aadd_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[str]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        await self._ensure_async_connection_ready()
        vectors, id_iterator = self._prepare_documents(documents, ids)
        ids_: list[str] = []
        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_data: dict[str, Value] = {
                "vector": cast(list[Value], vector),
                "text": doc.page_content,
                "metadata": doc.metadata,  # pyright: ignore[reportUnknownMemberType]
            }
            if doc_id is not None:
                record_id = RecordID(self.table, doc_id)
                inserted = await self.async_connection.upsert(record_id, doc_data)
            else:
                inserted = await self.async_connection.insert(self.table, doc_data)
            if isinstance(inserted, list):
                for record in inserted:
                    ids_.append(extract_id(record))
            elif isinstance(inserted, dict):
                ids_.append(extract_id(inserted))
        return ids_

    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        if ids is not None:
            for _id in ids:
                _ = self.connection.delete(RecordID(self.table, _id))
        else:
            _ = self.connection.delete(self.table)

    @override
    async def adelete(self, ids: list[str] | None = None, **kwargs: Any) -> None:  # pyright: ignore[reportAny, reportExplicitAny]
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        await self._ensure_async_connection_ready()
        if ids is not None:
            coroutines = [
                self.async_connection.delete(RecordID(self.table, _id)) for _id in ids
            ]
            _ = await asyncio.gather(*coroutines)
        else:
            _ = await self.async_connection.delete(self.table)

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        query_results = self.connection.query(
            GET_BY_ID_QUERY,
            {"table": self.table, "ids": list(ids)},
        )
        if not isinstance(query_results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_documents(ids, query_results)

    @override
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        await self._ensure_async_connection_ready()
        query_results = await self.async_connection.query(
            GET_BY_ID_QUERY,
            {"table": self.table, "ids": list(ids)},
        )
        if not isinstance(query_results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_documents(ids, query_results)

    def _build_search_query(
        self,
        vector: list[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: CustomFilter | None = None,
    ) -> tuple[str, QueryArgs]:
        args: QueryArgs = {
            "table": self.table,
            "vector": cast(list[Value], vector),
            "k": k,
            "score_threshold": score_threshold,
        }

        # build additional filter criteria
        custom_filter_str = ""
        if custom_filter:
            for key in custom_filter:
                args[key] = custom_filter[key]
                custom_filter_str += f"and metadata.{key} = ${key} "

        query = SEARCH_QUERY.format(k=k, custom_filter_str=custom_filter_str)
        return query, args

    def _similarity_search_with_score_by_vector(
        self,
        vector: list[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: CustomFilter | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        query, args = self._build_search_query(
            vector, k, score_threshold, custom_filter
        )
        results = self.connection.query(query, args)
        if not isinstance(results, list):
            raise ValueError(
                f"Invalid query results, expected a list. Result: {results}"
            )
        return self._parse_results(results)

    async def _asimilarity_search_with_score_by_vector(
        self,
        vector: list[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: CustomFilter | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        await self._ensure_async_connection_ready()
        query, args = self._build_search_query(
            vector, k, score_threshold, custom_filter
        )
        results = await self.async_connection.query(query, args)
        if not isinstance(results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_results(results)

    @override
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        custom_filter: CustomFilter | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[Document]:
        vector = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                vector=vector, k=k, custom_filter=custom_filter
            )
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[Document]:
        vector = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in await self._asimilarity_search_with_score_by_vector(
                vector=vector,
                k=k,
                **kwargs,  # pyright: ignore[reportAny]
            )
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[tuple[Document, float]]:
        vector = self.embedding.embed_query(query)
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                vector=vector,
                k=k,
                **kwargs,  # pyright: ignore[reportAny]
            )
        ]

    @override
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[tuple[Document, float]]:
        vector = self.embedding.embed_query(query)
        results: list[tuple[Document, float]] = []
        for doc, similarity, _ in await self._asimilarity_search_with_score_by_vector(
            vector=vector,
            k=k,
            **kwargs,  # pyright: ignore[reportAny]
        ):
            results.append((doc, similarity))
        return results

    ### ADDITIONAL OPTIONAL SEARCH METHODS BELOW ###

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[Document]:
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                vector=embedding,
                k=k,
                **kwargs,  # pyright: ignore[reportAny]
            )
        ]

    @override
    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> list[Document]:
        return [
            doc
            for doc, _, _ in await self._asimilarity_search_with_score_by_vector(
                vector=embedding,
                k=k,
                **kwargs,  # pyright: ignore[reportAny]
            )
        ]

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: CustomFilter | None = None,
        score_threshold: float = -1.0,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[Document]:
        vector = self.embedding.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            vector,
            k,
            fetch_k,
            lambda_mult,
            custom_filter=custom_filter,
            score_threshold=score_threshold,
            **kwargs,
        )
        return docs

    @override
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: CustomFilter | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> list[Document]:
        vector = self.embedding.embed_query(query)
        docs = await self.amax_marginal_relevance_search_by_vector(
            vector, k, fetch_k, lambda_mult, custom_filter=custom_filter, **kwargs
        )
        return docs

    def _similarity_search_by_vector_with_score(
        self,
        vector: list[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: CustomFilter | None = None,
    ) -> list[tuple[Document, float, list[float]]]:
        query, args = self._build_search_query(
            vector, k, score_threshold, custom_filter
        )
        results = self.connection.query(query, args)
        if not isinstance(results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_results(results)

    @staticmethod
    def _filter_documents_from_result(
        search_result: list[tuple[Document, float, list[float]]],
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        # extract only document from result
        docs = [sub[0] for sub in search_result]
        # extract only embedding from result
        vector = [sub[-1] for sub in search_result]

        mmr_selected = maximal_marginal_relevance(
            np.array(vector, dtype=np.float32),
            vector,
            k=k,
            lambda_mult=lambda_mult,
        )

        return [docs[i] for i in mmr_selected]

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: CustomFilter | None = None,
        score_threshold: float = -1.0,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[Document]:
        result = self._similarity_search_by_vector_with_score(
            embedding,
            fetch_k,
            custom_filter=custom_filter,
            score_threshold=score_threshold,
        )
        return self._filter_documents_from_result(result, k, lambda_mult)

    @override
    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: CustomFilter | None = None,
        **kwargs: Any,  # pyright: ignore[reportAny, reportExplicitAny]
    ) -> list[Document]:
        result = await self._asimilarity_search_with_score_by_vector(
            embedding,
            fetch_k,
            custom_filter=custom_filter,
            **kwargs,  # pyright: ignore[reportAny]
        )
        return self._filter_documents_from_result(result, k, lambda_mult)
