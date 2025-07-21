"""SurrealDB vector stores."""

from __future__ import annotations

import asyncio
from dataclasses import KW_ONLY, dataclass, field
from typing import (
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from surrealdb import (
    AsyncHttpSurrealConnection,
    AsyncWsSurrealConnection,
    BlockingHttpSurrealConnection,
    BlockingWsSurrealConnection,
    RecordID,
)

SurrealConnection = Union[BlockingWsSurrealConnection, BlockingHttpSurrealConnection]
SurrealAsyncConnection = Union[AsyncWsSurrealConnection, AsyncHttpSurrealConnection]
CustomFilter = dict[str, Union[str, bool, float, int]]
QueryArgs = dict[str, Union[int, float, str, list[float]]]

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
    metadata: dict[str, Any] = field(default_factory=dict)

    def into(self) -> Document:
        return Document(
            id=self.id.id,
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
    ) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        self.embedding = embedding
        self.table = table
        self.index_name = index_name
        self.connection = connection
        self.async_connection = async_connection
        if embedding_dimension is not None:
            self.embedding_dimension = embedding_dimension
        else:
            self.embedding_dimension = len(self.embedding.embed_query("foo"))
        self._ensure_index()

    def _ensure_index(self) -> None:
        query = DEFINE_INDEX.format(
            index_name=self.index_name,
            table=self.table,
            embedding_dimension=self.embedding_dimension,
        )
        self.connection.query(query)

    @staticmethod
    def _parse_documents(ids: Sequence[str], results: list[dict]) -> list[Document]:
        docs = {}
        for x in results:
            doc = SurrealDocument(
                text=x.pop("text"), vector=x.pop("vector"), **x
            ).into()
            docs[doc.id] = doc
        # sort docs in the same order as the passed in IDs
        result: list[Document] = []
        for key in ids:
            d = docs.get(str(key))
            if d is not None:
                result.append(d)
        return result

    @staticmethod
    def _parse_results(
        results: list[dict],
    ) -> list[tuple[Document, float, list[float]]]:
        parsed = []
        for raw in results:
            vector = raw.pop("vector")
            parsed.append(
                (
                    SurrealDocument(
                        text=raw.pop("text"),
                        vector=vector,
                        **raw,
                    ).into(),
                    raw["similarity"],
                    vector,
                ),
            )
        return parsed

    @classmethod
    def from_texts(
        cls: Type[SurrealDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        connection: SurrealConnection,
        table: str = "documents",
        index_name: str = "documents_vector_index",
        embedding_dimension: int | None = None,
        **kwargs: Any,
    ) -> SurrealDBVectorStore:
        store = cls(
            embedding=embedding,
            connection=connection,
            table=table,
            index_name=index_name,
            embedding_dimension=embedding_dimension,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @classmethod
    async def afrom_texts(
        cls: Type[SurrealDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        connection: SurrealConnection,
        async_connection: SurrealAsyncConnection | None = None,
        table: str = "documents",
        index_name: str = "documents_vector_index",
        embedding_dimension: int | None = None,
        **kwargs: Any,
    ) -> SurrealDBVectorStore:
        store = cls(
            embedding=embedding,
            connection=connection,
            async_connection=async_connection,
            table=table,
            index_name=index_name,
            embedding_dimension=embedding_dimension,
        )
        await store.aadd_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def _prepare_documents(
        self, documents: List[Document], ids: Optional[List[str]]
    ) -> tuple[List[List[float]], Iterator[Optional[str]]]:
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        return vectors, id_iterator

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the store."""
        vectors, id_iterator = self._prepare_documents(documents, ids)
        ids_ = []
        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_data = {
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            if doc_id is not None:
                record_id = RecordID(self.table, doc_id)
                inserted = self.connection.upsert(record_id, doc_data)
            else:
                inserted = self.connection.insert(self.table, doc_data)
            if isinstance(inserted, list):
                for record in inserted:
                    ids_.append(record["id"].id)
            else:
                ids_.append(inserted["id"].id)
        return ids_

    async def aadd_documents(
        self, documents: List[Document], ids: Optional[List[str]] = None, **kwargs: Any
    ) -> List[str]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        vectors, id_iterator = self._prepare_documents(documents, ids)
        ids_ = []
        coroutines = []
        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_data = {
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            if doc_id is not None:
                record_id = RecordID(self.table, doc_id)
                coroutines.append(self.async_connection.upsert(record_id, doc_data))
            else:
                coroutines.append(self.async_connection.insert(self.table, doc_data))
        results = await asyncio.gather(*coroutines)
        for inserted in results:
            if isinstance(inserted, list):
                for record in inserted:
                    ids_.append(record["id"].id)
            elif isinstance(inserted, dict):
                ids_.append(inserted["id"].id)
        return ids_

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if ids is not None:
            for _id in ids:
                self.connection.delete(RecordID(self.table, _id))
        else:
            self.connection.delete(self.table)

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        if ids is not None:
            coroutines = [
                self.async_connection.delete(RecordID(self.table, _id)) for _id in ids
            ]
            await asyncio.gather(*coroutines)
        else:
            await self.async_connection.delete(self.table)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        query_results = self.connection.query(
            GET_BY_ID_QUERY,
            {"table": self.table, "ids": ids},
        )
        if not isinstance(query_results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_documents(ids, query_results)

    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        query_results = await self.async_connection.query(
            GET_BY_ID_QUERY,
            {"table": self.table, "ids": ids},
        )
        if not isinstance(query_results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_documents(ids, query_results)

    def _build_search_query(
        self,
        vector: List[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: Optional[CustomFilter] = None,
    ) -> tuple[str, QueryArgs]:
        args: QueryArgs = {
            "table": self.table,
            "vector": vector,
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
        vector: List[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: Optional[CustomFilter] = None,
    ) -> List[tuple[Document, float, List[float]]]:
        query, args = self._build_search_query(
            vector, k, score_threshold, custom_filter
        )
        results = self.connection.query(query, args)
        if not isinstance(results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_results(results)

    async def _asimilarity_search_with_score_by_vector(
        self,
        vector: List[float],
        k: int = 4,
        score_threshold: float = -1.0,
        custom_filter: Optional[CustomFilter] = None,
    ) -> List[tuple[Document, float, List[float]]]:
        if self.async_connection is None:
            raise ValueError("No async connection provided")
        query, args = self._build_search_query(
            vector, k, score_threshold, custom_filter
        )
        results = await self.async_connection.query(query, args)
        if not isinstance(results, list):
            raise ValueError("Invalid query results, expected a list")
        return self._parse_results(results)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        custom_filter: Optional[CustomFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
        vector = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                vector=vector, k=k, custom_filter=custom_filter
            )
        ]

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        vector = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in await self._asimilarity_search_with_score_by_vector(
                vector=vector, k=k, **kwargs
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        vector = self.embedding.embed_query(query)
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                vector=vector, k=k, **kwargs
            )
        ]

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        vector = self.embedding.embed_query(query)
        results = []
        for doc, similarity, _ in await self._asimilarity_search_with_score_by_vector(
            vector=vector, k=k, **kwargs
        ):
            results.append((doc, similarity))
        return results

    ### ADDITIONAL OPTIONAL SEARCH METHODS BELOW ###

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                vector=embedding, k=k, **kwargs
            )
        ]

    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [
            doc
            for doc, _, _ in await self._asimilarity_search_with_score_by_vector(
                vector=embedding, k=k, **kwargs
            )
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: Optional[CustomFilter] = None,
        score_threshold: float = -1.0,
        **kwargs: Any,
    ) -> List[Document]:
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

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: Optional[CustomFilter] = None,
        **kwargs: Any,
    ) -> List[Document]:
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
        custom_filter: Optional[CustomFilter] = None,
    ) -> list[tuple[Document, float, list[float]]]:
        if self.connection is None:
            raise ValueError("No connection provided")
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

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: Optional[CustomFilter] = None,
        score_threshold: float = -1.0,
        **kwargs: Any,
    ) -> list[Document]:
        result = self._similarity_search_by_vector_with_score(
            embedding,
            fetch_k,
            custom_filter=custom_filter,
            score_threshold=score_threshold,
        )
        return self._filter_documents_from_result(result, k, lambda_mult)

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        custom_filter: Optional[CustomFilter] = None,
        **kwargs: Any,
    ) -> list[Document]:
        result = await self._asimilarity_search_with_score_by_vector(
            embedding, fetch_k, custom_filter=custom_filter, **kwargs
        )
        return self._filter_documents_from_result(result, k, lambda_mult)
