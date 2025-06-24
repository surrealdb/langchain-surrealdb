---
title: Building a simple GenAI chatbot using GraphRAG
sub_title: SurrealDB + LangChain
author: Martin Schaer <martin.schaer@surrealdb.com>
theme:
  name: surreal
---

What is traditional RAG
===

Retrieval-Augmented Generation (RAG) is an AI technique that enhances the capabilities of large language models (LLMs) by allowing them to retrieve relevant information from a knowledge base *before* generating a response. It uses vector similarity search to find the most relevant document chunks, which are then provided as additional context to the LLM, enabling the LLM to produce more accurate and grounded responses.

What is GraphRAG
===

Graph RAG is an advanced technique. It leverages the structured, interconnected nature of knowledge graphs to provide the LLM with a richer, more contextualized understanding of the information, leading to more accurate, coherent, and less "hallucinated" responses.

<!-- end_slide -->

Flow overview
===

1. ingest data (categorized health symptoms and common treatments)
2. ask the user for their symptom
3. find relevant documents in the DB
4. find related common treatments

<!-- end_slide -->

First step: ingest the data
===

<!-- column_layout: [1, 1] -->

<!-- column: 0 -->

For this example we have a YAML file with categorised symptoms and their common treatments.

We want to store this in a vector store so we can query it using vector similarity search.

We also want to represent the data relations in a graph store, so we can run graph queries to retrieve those relationships (e.g. treatments related to symptoms).

<!-- column: 1 -->
```yaml
- category: General Symptoms
  symptoms:
    - name: Fever
      description: Elevated body temperature, usually above 100.4°F (38°C).
      medical_practice: General Practice, Internal Medicine, Pediatrics
      possible_treatments:
        - Antipyretics (e.g., ibuprofen, acetaminophen)
        - Rest
        - Hydration
        - Treating the underlying cause
```

<!-- end_slide -->

Let's instantiate the following LangChain components:

- **Vector Store**
- **Graph Store**
- **OllamaEmbeddings**

...and create a SurrealDB connection:

```python
# DB connection
conn = Surreal(url)
conn.signin({"username": user, "password": password})
conn.use(ns, db)

# Vector Store
vector_store = SurrealDBVectorStore(
    OllamaEmbeddings(model="llama3.2"),
    conn
)

# Graph Store
graph_store = SurrealDBGraph(conn)
```

Note that the `SurrealDBVectorStore` is instantiated with `OllamaEmbeddings`. This LLM model will be used when inserting documents to generate their embeddings vector.

<!-- end_slide -->

## Now we are ready to populate the vector store

```python
# Parsing the YAML into a Symptoms dataclass
with open("./symptoms.yaml", "r") as f:
    symptoms = yaml.safe_load(f)
    assert isinstance(symptoms, list), "failed to load symptoms"
    for category in symptoms:
        parsed_category = Symptoms(category["category"], category["symptoms"])
        for symptom in parsed_category.symptoms:
            parsed_symptoms.append(symptom)
            symptom_descriptions.append(
                Document(
                    page_content=symptom.description.strip(),
                    metadata=asdict(symptom),
                )
            )

# This calculates the embeddings and inserts the documents in the DB
vector_store.add_documents(symptom_descriptions)
```

<!-- end_slide -->

## Stitching the graph together

```python
# Find nodes and edges (Treatment -> Treats -> Symptom)
for idx, category_doc in enumerate(symptom_descriptions):
    # Nodes
    treatment_nodes = {}
    symptom = parsed_symptoms[idx]
    symptom_node = Node(id=symptom.name, type="Symptom", properties=asdict(symptom))
    for x in symptom.possible_treatments:
        treatment_nodes[x] = Node(id=x, type="Treatment", properties={"name": x})
    nodes = list(treatment_nodes.values())
    nodes.append(symptom_node)

    # Edges
    relationships = [
        Relationship(source=treatment_nodes[x], target=symptom_node, type="Treats")
        for x in symptom.possible_treatments
    ]
    graph_documents.append(
        GraphDocument(nodes=nodes, relationships=relationships, source=category_doc)
    )

# Store the graph
graph_store.add_graph_documents(graph_documents, include_source=True)
```

<!-- end_slide -->

Data ready, let's chat
===

LangChain provides different [chat models](https://python.langchain.com/docs/integrations/chat/). We are going to use `ChatOllama` with `llama3.2` to generate a graph query and to explain the result in natural language.

```python
chat_model = ChatOllama(model="llama3.2", temperature=0)
```

To generate the graph query based on the user's prompt, we need to instantiate a QA (Questioning and Answering:) Chain component. In this case we are using `SurrealDBGraphQAChain`.

But before quering the graph, we need to find in our vector store the symptoms by doing a similarity search based on the user's prompt.

```python
query = click.prompt(
    click.style("\nWhat are your symptoms?", fg="green"), type=str
)

# -- Find relevant docs
docs = vector_search(query, vector_store, k=3)
symptoms = get_document_names(docs)

# -- Query the graph
chain = SurrealDBGraphQAChain.from_llm(
    chat_model,
    graph=graph_store,
    verbose=verbose,
    query_logger=query_logger,
)
ask(f"what medical practices can help with {symptoms}", chain)
ask(f"what treatments can help with {symptoms}", chain)
```

<!-- end_slide -->

Running
===

First we start the DB:

```bash
surreal start -u root -p root
```

Second, we ingest the data:

```bash
just examples-graph ingest
# or
poetry run run ingest
```

Third, we start the CLI chat:

```bash
just examples-graph chat
# verbose logs enables
just examples-graph chat --verbose
# or directly with poetry
poetry run run chat
# and
poetry run run chat --verbose
```

Running the program will look like this:

```
What are your symptoms?: i have a runny nose and itchy eyes
```

The script tries marginal relevance and similarity searches in the vector store to compare the results, which helps to choose the right one for your specific use case.

```
max_marginal_relevance_search:
- Stuffy nose due to inflamed nasal passages or a dripping nose with mucus discharge.
- An uncomfortable sensation that makes you want to scratch, often without visible skin changes.
- Feeling lightheaded, unsteady, or experiencing a sensation that the room is spinning.

similarity_search_with_score
- [40%] Stuffy nose due to inflamed nasal passages or a dripping nose with mucus discharge.
- [33%] Feeling lightheaded, unsteady, or experiencing a sensation that the room is spinning.
- [32%] Pain, irritation, or scratchiness in the throat, often made worse by swallowing.
```

Then, the QA chain will generate and run a graph query behind the scenes, and generate the responses.

This script is asking our AI two questions based on the user's symptoms:
- Question: what medical practices can help with Nasal Congestion/Runny Nose, Dizziness/Vertigo, Sore Throat
- Question: what treatments can help with Nasal Congestion/Runny Nose, Dizziness/Vertigo, Sore Throat

For the first question the QA chain component generated this graph query:
```
SELECT <-relation_Attends<-graph_Practice as practice FROM graph_Symptom WHERE name IN ["Nasal Congestion/Runny Nose", "Dizziness/Vertigo", "Sore Throat"];
```

And for the second:
```
SELECT <-relation_Treats<-graph_Treatment as treatment FROM graph_Symptom WHERE name IN ["Nasal Congestion/Runny Nose", "Dizziness/Vertigo", "Sore Throat"]
```

The results of these two queries are Python lists of dictionaries containing the treatment and medical practices names, which are fed to the LLM to generate a nice human readable answer:

    Here is a summary of the medical practices that can help with Nasal Congestion/Runny Nose, Dizziness/Vertigo, and Sore Throat:

    Several medical practices may be beneficial for individuals experiencing symptoms such as Nasal Congestion/Runny Nose, Dizziness/Vertigo, and Sore Throat. These include Neurology, ENT (Otolaryngology), General Practice, and Allergy & Immunology.

    Neurology specialists can provide guidance on managing conditions that affect the nervous system, which may be related to dizziness or vertigo. ENT (Otolaryngology) specialists focus on ear, nose, and throat issues, making them a good fit for addressing nasal congestion and runny nose symptoms. General Practice physicians offer comprehensive care for various health concerns, including those affecting the respiratory system.

    Allergy & Immunology specialists can help diagnose and treat allergies that may contribute to Nasal Congestion/Runny Nose, as well as provide immunological support for overall health.

And regarding the possible treatments:

    Here is a summary of the treatments that can help with Nasal Congestion/Runny Nose, Dizziness/Vertigo, and Sore Throat:

    The following treatments have been found to be effective in alleviating symptoms:

    - Vestibular rehabilitation
    - Hydration
    - Medications to reduce nausea or dizziness
    - Antihistamines (for allergies)
    - Decongestants (oral or nasal sprays)
    - Saline nasal rinses
    - Humidifiers
    - Throat lozenges/sprays
    - Treating underlying cause (e.g., cold, allergies)
    - Pain relievers (e.g., acetaminophen, ibuprofen)
    - Warm salt water gargles

<!-- end_slide -->
