import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

INSTRUCTION_PROMPT_INFER_KEYWORDS = """
---Role---
You are a helpfull assistant tasked with identifying keywords in texts. You can
also identify texts that don't relate to any important keyword.

---Goal---
Given the text, list keywords. Keywords are general concepts or themes.
"""

INPUT_PROMPT_INFER_KEYWORDS = r"""
- Keywords are general concepts or themes. They are broad categories, not specific at all.
- Idenfify if the text has a relevant theme, if not, do not provide keywords
- Output only the keywords as a comma-separated list
- Do not explain your answer
- Your output shoud be emtpy if the text is not providing useful information

---Examples---

Example 1:
Text: "Let's go to italy during the holidays"
Output: "italy,holiday,travel"

Example 2:
Text: "ugh.. :("
Output: ""

Example 3:
Text: "I have plans on sunday"
Output: "schedule,sunday"

Example 4:
Text: "Hello, how are you?"
Output: ""

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
