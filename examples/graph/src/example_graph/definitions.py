from dataclasses import dataclass

from typing_extensions import cast


@dataclass
class Symptom:
    name: str
    description: str
    category: str
    medical_practice: list[str]
    possible_treatments: list[str]


class Symptoms:
    def __init__(self, category: str, symptoms: list[dict[str, str | list[str]]]):
        self.category: str = category
        self.symptoms: list[Symptom] = [
            Symptom(
                name=str(x.get("name", "")),
                description=str(x.get("description", "")),
                category=category,
                medical_practice=[
                    y.strip() for y in str(x.get("medical_practice", "")).split(",")
                ],
                possible_treatments=cast(list[str], x.get("possible_treatments", [])),
            )
            for x in symptoms
        ]
