from datetime import datetime
from typing import Any

from langchain_core.messages import BaseMessage


def normalize_content(msg_content: str | list[str | dict[str, Any]]) -> str:  # pyright: ignore[reportExplicitAny]
    if isinstance(msg_content, str):
        return msg_content
    else:
        return "\n".join(str(msg_content))


def parse_time(message_time: str | None) -> datetime:
    if message_time is None:
        dt = datetime.now()
    else:
        try:
            dt = datetime.strptime(message_time, "%d/%m/%Y, %H:%M:%S")
        except ValueError:
            dt = datetime.now()
    return dt


def format_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        return f"{dt.isoformat(timespec='seconds')}+00:00"
    return dt.isoformat(timespec="seconds")


def get_message_timestamp_and_sender(msg: BaseMessage) -> tuple[datetime, str]:
    events = msg.additional_kwargs.get("events", [])  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    sender = str(msg.additional_kwargs.get("sender", ""))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    if events:
        if isinstance(events, list) and isinstance(events[0], dict):
            tmp = events[0]["message_time"]  # pyright: ignore[reportUnknownVariableType]
        else:
            tmp = None
        if isinstance(tmp, datetime):
            return tmp, sender
        else:
            return parse_time(str(tmp)), sender  # pyright: ignore[reportUnknownArgumentType]
    else:
        return datetime.now(), sender
