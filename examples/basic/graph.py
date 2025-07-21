import time

from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from surrealdb import Surreal

from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

conn = Surreal("ws://localhost:8000/rpc")
conn.signin({"username": "root", "password": "root"})
conn.use("langchain", "demo")
vector_store = SurrealDBVectorStore(OllamaEmbeddings(model="all-minilm:22m"), conn)
graph_store = SurrealDBGraph(conn)

vector_store.delete()
graph_store.delete_nodes()

# ------------------------------------------------------------------------------
# -- Vector
doc1 = Document(
    page_content="SurrealDB is the ultimate multi-model database for AI applications",
    metadata={"key": "sdb"},
)
doc2 = Document(
    page_content="Surrealism is an artistic and cultural movement that emerged in the early 20th century",
    metadata={"key": "surrealism"},
)
vector_store.add_documents(documents=[doc1, doc2], ids=["1", "2"])

# ------------------------------------------------------------------------------
# -- Graph

# Documents nodes
node_sdb = Node(id="sdb", type="Document")
node_surrealism = Node(id="surrealism", type="Document")

# People nodes
node_martin = Node(id="martin", type="People", properties={"name": "Martin"})
node_tobie = Node(id="tobie", type="People", properties={"name": "Tobie"})
node_max = Node(id="max", type="People", properties={"name": "Max Ernst"})

# Edges
graph_documents = [
    GraphDocument(
        nodes=[node_martin, node_tobie, node_sdb],
        relationships=[
            Relationship(source=node_martin, target=node_sdb, type="KnowsAbout"),
            Relationship(source=node_tobie, target=node_sdb, type="KnowsAbout"),
        ],
        source=doc1,
    ),
    GraphDocument(
        nodes=[node_max, node_surrealism],
        relationships=[
            Relationship(source=node_max, target=node_surrealism, type="KnowsAbout")
        ],
        source=doc2,
    ),
]

graph_store.add_graph_documents(graph_documents)

# ------------------------------------------------------------------------------
# -- LLM
model = OllamaLLM(model="llama3.2", temperature=1, verbose=True)

# Let's retrieve information about these 2 topics
queries = ["database", "surrealism"]
for q in queries:
    print(f'\n----------------------------------\nTopic: "{q}"\nVector search:')
    results = vector_store.similarity_search_with_score(query=q, k=2)
    for doc, score in results:
        print(f"• [{score:.0%}]: {doc.page_content}")
    top_match = results[0][0]

    # Graph query
    res = graph_store.query(
        """
        SELECT <-relation_KnowsAbout<-graph_People as people
        FROM type::thing("graph_Document", $doc_key)
        FETCH people
        """,
        {"doc_key": top_match.metadata.get("key")},
    )
    people = [x.get("name") for x in res[0].get("people", [])]

    print(f"\nGraph result: {people}")

    # Template for the LLM
    template = """
    You are a young, energetic database developer in your last 20s, who loves to
    talk tech, and who's also very geeky.
    Use the following pieces of retrieved context to answer the question.
    Use four sentences maximum and keep the answer concise.
    Try to be funny with a play on words.

    Context: {context}. People who know about this: {people}.

    Question: Explain "{topic}", summarize the context provided, and tell me
    who I can ask for more information.

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    answer = chain.invoke(
        {"context": top_match.page_content, "people": people, "topic": q}
    )
    print(f"\nLLM answer:\n===========\n{answer}")
    time.sleep(4)

print("\nBye!")
