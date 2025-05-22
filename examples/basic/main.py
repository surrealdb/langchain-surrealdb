from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from surrealdb import Surreal

from langchain_surrealdb.vectorstores import SurrealDBVectorStore

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
    print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")  # noqa: T201
