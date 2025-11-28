import json
import logging
import os
import re
import zipfile
from collections.abc import Iterator
from datetime import datetime

from langchain_core.chat_loaders import BaseChatLoader
from langchain_core.chat_sessions import ChatSession
from langchain_core.messages import HumanMessage
from typing_extensions import override

logger = logging.getLogger(__name__)


class InstagramChatLoader(BaseChatLoader):
    """Load `Instagram` conversations from a dump zip file or directory."""

    def __init__(self, path: str):
        """Initialize the InstagramChatLoader.

        Args:
            path (str): Path to the exported Instagram chat
                zip directory, folder, or file.
        """
        self.path: str = path
        ignore_lines = [".* sent an attachment.", "Liked a message"]
        self._ignore_lines: re.Pattern[str] = re.compile(
            r"(" + "|".join([line for line in ignore_lines]) + r")",
            flags=re.IGNORECASE,
        )

    def _load_single_chat_session(self, file_path: str) -> ChatSession:
        """Load a single chat session from a file.

        Args:
            file_path (str): Path to the chat file.

        Returns:
            ChatSession: The loaded chat session.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            json_parsed = json.load(file)  # pyright: ignore[reportAny]

        results: list[HumanMessage] = []
        messages = json_parsed.get("messages", [])  # pyright: ignore[reportAny]
        assert isinstance(messages, list)
        # sort by timestamp
        messages.sort(key=lambda x: x.get("timestamp_ms", 0))  # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
        for message in messages:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(message, dict):
                sender = message.get("sender_name")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                timestamp = datetime.fromtimestamp(
                    message.get("timestamp_ms", 0) / 1000  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
                )
                text = str(message.get("content", ""))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                if not self._ignore_lines.match(text.strip()):
                    results.append(
                        HumanMessage(
                            role=sender,
                            content=text,
                            additional_kwargs={
                                "sender": sender,
                                "events": [{"message_time": timestamp}],
                            },
                        )
                    )
        return ChatSession(messages=results)

    @staticmethod
    def _iterate_files(path: str) -> Iterator[str]:
        """Iterate over the files in a directory or zip file.

        Args:
            path (str): Path to the directory or zip file.

        Yields:
            str: The path to each file.
        """
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file == "message_1.json":
                        yield os.path.join(root, file)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zip_file:
                for file in zip_file.namelist():
                    if file == "message_1.json":
                        yield zip_file.extract(file)

    @override
    def lazy_load(self) -> Iterator[ChatSession]:
        """Lazy load the messages from the chat file and yield
        them as chat sessions.

        Yields:
            Iterator[ChatSession]: The loaded chat sessions.
        """
        yield self._load_single_chat_session(self.path)
