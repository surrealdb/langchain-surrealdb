from dataclasses import dataclass


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
                name=x.get("name", ""),
                description=x.get("description", ""),
                category=category,
                medical_practice=[
                    y.strip() for y in x.get("medical_practice", "").split(",")
                ],
                possible_treatments=x.get("possible_treatments", []),
            )
            for x in symptoms
        ]
