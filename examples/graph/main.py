from dataclasses import asdict, dataclass

import yaml
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from surrealdb import Surreal

from langchain_surrealdb.graph_qa.chain import SurrealDBGraphQAChain
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


@dataclass
class Symptom:
    name: str
    description: str
    category: str
    medical_practice: list[str]
    possible_treatments: list[str]


class Symptoms:
    def __init__(self, category: str, symptoms: list[dict]):
        self.category = category
        self.symptoms = [
            Symptom(
                name=x.get("name"),
                description=x.get("description"),
                category=category,
                medical_practice=[
                    y.strip() for y in x.get("medical_practice", "").split(",")
                ],
                possible_treatments=x.get("possible_treatments", []),
            )
            for x in symptoms
        ]


# -- Insert documents
symptom_descriptions: list[Document] = []
parsed_symptoms: list[Symptom] = []
with open("symptoms.yaml", "r") as f:
    symptoms = yaml.safe_load(f)
    assert isinstance(symptoms, list), "failed to load symptoms"
    for category in symptoms:
        parsed_category = Symptoms(category["category"], category["symptoms"])
        for symptom in parsed_category.symptoms:
            parsed_symptoms.append(symptom)
            symptom_descriptions.append(
                Document(
                    page_content=symptom.description.strip(), metadata=asdict(symptom)
                )
            )
vector_store.add_documents(symptom_descriptions)

# -- Generate graph
print("Generating graph...")
graph_documents = []
for idx, category_doc in enumerate(symptom_descriptions):
    practice_nodes = {}
    treatment_nodes = {}
    symptom = parsed_symptoms[idx]
    symptom_node = Node(id=symptom.name, type="Symptom", properties=asdict(symptom))
    for x in symptom.medical_practice:
        practice_nodes[x] = Node(id=x, type="Practice", properties={"name": x})
    for x in symptom.possible_treatments:
        treatment_nodes[x] = Node(id=x, type="Treatment", properties={"name": x})
    nodes = list(practice_nodes.values()) + list(treatment_nodes.values())
    nodes.append(symptom_node)
    relationships = [
        Relationship(source=practice_nodes[x], target=symptom_node, type="Attends")
        for x in symptom.medical_practice
    ] + [
        Relationship(source=treatment_nodes[x], target=symptom_node, type="Treats")
        for x in symptom.possible_treatments
    ]
    graph_documents.append(
        GraphDocument(nodes=nodes, relationships=relationships, source=category_doc)
    )
graph_store.add_graph_documents(graph_documents, include_source=True)
print("stored!")

# -- User query
# query = "i have a headache"
query = "i can't sleep"

# -- Vector search
print(f'\nSearch: "{query}"\n')  # noqa: T201
results = vector_store.max_marginal_relevance_search(
    query, k=3, fetch_k=20, score_threshold=0.3
)
print("max_marginal_relevance_search")
for doc in results:
    print(f"- {doc.page_content}")  # noqa: T201
results = vector_store.similarity_search_with_score(query, k=3)
print("similarity_search_with_score")
for doc, score in results:
    print(f"[similarity={score:.0%}] {doc.page_content}")  # noqa: T201

# -- Query Graph
chain = SurrealDBGraphQAChain.from_llm(chat_model, graph=graph_store, verbose=True)
response = chain.invoke(
    {"query": f"what medical practices can help with {results[0][0].metadata['name']}"}
)
print(response["result"])  # noqa: T201
print("\nAnswer:")  # noqa: T201
print(response["result"]["text"])  # noqa: T201

response = chain.invoke(
    {"query": f"what treatments can help with {results[0][0].metadata['name']}"}
)
print(response["result"])  # noqa: T201
print("\nAnswer:")  # noqa: T201
print(response["result"]["text"])  # noqa: T201
