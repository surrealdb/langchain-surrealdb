from dataclasses import asdict

import yaml
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document

from langchain_surrealdb.experimental.surrealdb_graph import SurrealDBGraph
from langchain_surrealdb.vectorstores import SurrealDBVectorStore

from .definitions import Symptom, Symptoms


def ingest(vector_store: SurrealDBVectorStore, graph_store: SurrealDBGraph) -> None:
    # -- Insert documents
    symptom_descriptions: list[Document] = []
    parsed_symptoms: list[Symptom] = []
    with open("./data/symptoms.yaml", "r") as f:
        symptoms = yaml.safe_load(f)  # pyright: ignore[reportAny]
        assert isinstance(symptoms, list), "failed to load symptoms"
        for category in symptoms:  # pyright: ignore[reportUnknownVariableType]
            parsed_category = Symptoms(category["category"], category["symptoms"])  # pyright: ignore[reportUnknownArgumentType]
            for symptom in parsed_category.symptoms:
                parsed_symptoms.append(symptom)
                symptom_descriptions.append(
                    Document(
                        page_content=symptom.description.strip(),
                        metadata=asdict(symptom),
                    )
                )
    _ = vector_store.add_documents(symptom_descriptions)

    # -- Generate graph
    print("Generating graph...")  # noqa: T201
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
        nodes = list(practice_nodes.values()) + list(treatment_nodes.values())  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        nodes.append(symptom_node)  # pyright: ignore[reportUnknownMemberType]
        relationships = [
            Relationship(source=practice_nodes[x], target=symptom_node, type="Attends")  # pyright: ignore[reportUnknownArgumentType]
            for x in symptom.medical_practice
        ] + [
            Relationship(source=treatment_nodes[x], target=symptom_node, type="Treats")  # pyright: ignore[reportUnknownArgumentType]
            for x in symptom.possible_treatments
        ]
        graph_documents.append(  # pyright: ignore[reportUnknownMemberType]
            GraphDocument(nodes=nodes, relationships=relationships, source=category_doc)  # pyright: ignore[reportUnknownArgumentType]
        )
    graph_store.add_graph_documents(graph_documents, include_source=True)  # pyright: ignore[reportUnknownArgumentType]
    print("stored!")  # noqa: T201
