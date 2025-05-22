from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama, OllamaEmbeddings
from surrealdb import Surreal

from langchain_surrealdb.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

model = "llama3.2"
chat_model = ChatOllama(model=model, temperature=0)

conn = Surreal("ws://localhost:8000/rpc")
conn.signin({"username": "root", "password": "root"})
conn.use("langchain", "example-graph")
vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="llama3.2"), conn)
vector_store.delete()
graph_store = SurrealDBGraph(conn)
graph_store.delete_nodes()

# -- Insert documents
docs = []
with open("knowledge.txt", "r") as f:
    for line in f:
        docs.append(Document(page_content=line.strip()))
vector_store.add_documents(docs)

# -- Generate graph
llm_transformer = LLMGraphTransformer(
    llm=chat_model,
    # TODO: figure this out
    # allowed_nodes=["Person", "Ship", "Location", "Organization", "Component"],
    # allowed_relationships=[
    #     ("Person", "LOCATED_AT", "Location"),
    #     ("Organization", "LOCATED_AT", "Location"),
    #     ("Person", "MEMBER_OF", "Organization"),
    #     ("Ship", "HAS", "Component"),
    #     ("Organization", "OWNS", "Ship"),
    #     ("Organization", "OWNS", "Component"),
    # ],
    # prompt=ChatPromptTemplate("Every object in a ship is a Component"),
    # strict_mode=False,
)
graph_documents = llm_transformer.convert_to_graph_documents(docs)
graph_store.add_graph_documents(graph_documents, include_source=True)

# -- Query
q = "how can i fix my ship"
print(f'\nSearch: "{q}"\n')  # noqa: T201
results = vector_store.similarity_search_with_score(query=q, k=4)
for doc, score in results:
    print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201
