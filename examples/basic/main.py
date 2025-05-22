from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from surrealdb import Surreal

from langchain_surrealdb.vectorstores import SurrealDBVectorStore

conn = Surreal("ws://localhost:8000/rpc")
conn.signin({"username": "root", "password": "root"})
conn.use("langchain", "demo")
vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)

_url = "https://surrealdb.com"
d1 = Document(page_content="foo", metadata={"source": _url})
d2 = Document(page_content="SurrealDB", metadata={"source": _url})
d3 = Document(page_content="bar", metadata={"source": "https://example.com"})
d4 = Document(page_content="this is surreal", metadata={"source": _url})

vector_store.add_documents(documents=[d1, d2, d3, d4], ids=["1", "2", "3", "4"])

q = "surreal"
filter = {"source": _url}
print(f'\nSearch: "{q}" filter {filter}\n')  # noqa: T201
results = vector_store.similarity_search_with_score(query=q, k=2, custom_filter=filter)
for doc, score in results:
    print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201
