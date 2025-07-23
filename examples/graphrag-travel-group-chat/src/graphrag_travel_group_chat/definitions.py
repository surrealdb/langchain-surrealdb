from dataclasses import dataclass
from datetime import datetime


@dataclass
class Chunk:
    senders: set[str]
    content: str
    timestamp: datetime
