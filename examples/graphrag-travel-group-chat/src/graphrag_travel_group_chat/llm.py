import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

INSTRUCTION_PROMPT_INFER_KEYWORDS = """
---Role---
You are a helpfull assistant tasked with identifying themes and topics in chat
messages.

---Goal---
Given a series of chat messages, list the keywords that represent the main
themes and topics in the messages.
"""

INPUT_PROMPT_INFER_KEYWORDS = r"""
- Keywords are general concepts or themes. They should be as broad as possible.
- Output only the keywords as a comma-separated list.
- Your output can be empty if the text does not provide useful information.
- Do not explain your answer.

---Examples---

Example 1:
Text: "Martin: Let's go to italy during the holidays"
Output: italy,holiday,travel

Example 2:
Text: "Chloe: ugh.. :("
Output:

Example 3:
Text: "Liam: I have plans on sunday"
Output: schedule,sunday

Example 4:
Text: "Ben: Hello, how are you?\nChloe: I'm doing fine!"
Output:

---Real Data---
Text: {text}
Output:
"""  # noqa: E501


def infer_keywords(text: str) -> set[str]:
    chat_model = ChatOllama(model="llama3.2", temperature=0)
    prompt = ChatPromptTemplate(
        [
            ("system", INSTRUCTION_PROMPT_INFER_KEYWORDS),
            ("user", INPUT_PROMPT_INFER_KEYWORDS),
        ]
    )
    chain = prompt | chat_model
    res = chain.invoke({"text": text})
    logger.debug(f"_infer_keywords: {res}")
    if isinstance(res.content, str):
        return set([x.strip().lower() for x in res.content.split(",")])
    else:
        return set([x.strip().lower() for x in str(res.content).split(",")])
