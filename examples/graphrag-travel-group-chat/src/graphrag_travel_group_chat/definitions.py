from dataclasses import dataclass


@dataclass
class MessageKeywords:
    id: str
    sender: str
    keywords: set[str]
