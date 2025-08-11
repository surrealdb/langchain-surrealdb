from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

INSTRUCTION_PROMPT_INFER_KEYWORDS = """
---Role---
You are a helpfull assistant tasked with identifying themes and topics in chat
messages.

---Goal---
Given a series of chat messages, list the keywords that represent the main
themes and topics in the messages.
"""

INPUT_PROMPT_INFER_KEYWORDS = r"""
- Include at least 1 keyword that describes the feeling or sentiment of the
converation
- Keywords are general concepts or themes. They should be as broad as possible.
- Output only the keywords as a comma-separated list.
- Do not explain your answer.

{additional_instructions}

---Examples---

Example 1:
Text: "[2025-07-10 22:21:59] Martin: Let's go to italy during the holidays"
Output: italy,holiday,travel

Example 2:
Text: "[2025-04-07 14:14:26] Chloe: ugh.. :("
Output:

Example 3:
Text: "[2025-03-19 08:01:35] Liam: I have plans on sunday"
Output: schedule,sunday

Example 4:
Text: "[2025-01-19 10:30:16] Ben: Hello, how are you?\nChloe: I'm doing fine!"
Output:

---Real Data---
Text: {text}
Output:
"""

ADDITIONAL_INSTRUCTIONS = r"""
"""

SUMMARY_FROM_CHATS_PROMPT = r"""
---Role---
You are a helpful assistant tasked with generating summaries from chat
extracts.

---Goal---
Provide a complete summary that responds to the user's query based on the
provided chat extracts.

---Instructions---
- Include all the details from the chat extracts in your summary.
- If the texts are contradictory, use the ones that have more details to form
    your answer.
- Base your answer on the provided context, not on general knowledge.
- Don't mention how you know the answer.
- Do not ask questions.

---Chat extracts---
{context}
"""

SUMMARIZE_ANSWER_PROMPT = """
---Role---
You are a helpful assistant tasked with answering user questions based on the
user's conversations with other people.

---Goal---
Provide an answer to the user's query based on the provided context.

---Instructions---
- If the user says "we", he means himself and his/her chat friends, it does not
    include yourself.
- You know the user from before.
- If the texts are contradictory, use the ones that have more details to form
    your answer.
- Don't mention how you know the answer, or previous conversations.
- Do not ask questions.

---Context---
{context}
"""


def infer_keywords(text: str, all_keywords: list[str] | None) -> set[str]:
    chat_model = ChatOllama(model="llama3.2", temperature=0)
    prompt = ChatPromptTemplate(
        [
            ("system", INSTRUCTION_PROMPT_INFER_KEYWORDS),
            ("user", INPUT_PROMPT_INFER_KEYWORDS),
        ]
    )
    chain = prompt | chat_model
    res = chain.invoke(
        {
            "text": text,
            "additional_instructions": ADDITIONAL_INSTRUCTIONS.format(
                all_keywords=all_keywords
            )
            if all_keywords
            else "",
        }
    )
    if isinstance(res.content, str):
        return set([x.strip().lower() for x in res.content.split(",")])
    else:
        return set([x.strip().lower() for x in str(res.content).split(",")])


def generate_answer_from_messages(chats: list[str], query: str, user_name: str) -> str:
    chat_model = ChatOllama(model="llama3.2", temperature=0.8)
    messages = [
        ("system", SUMMARY_FROM_CHATS_PROMPT),
        ("user", "My name is {user_name}. {query}"),
    ]
    prompt = ChatPromptTemplate(messages)
    chain = prompt | chat_model
    res = chain.invoke(
        {
            "user_name": user_name,
            "query": query,
            "context": [f"\n{txt}\n" for txt in chats],
        }
    )
    return str(res.content)


def summarize_answer(context: list[str], query: str, user_name: str) -> str:
    chat_model = ChatOllama(model="llama3.2", temperature=0.8)
    messages = [
        ("system", SUMMARIZE_ANSWER_PROMPT),
        ("user", "My name is {user_name}. {query}"),
    ]
    prompt = ChatPromptTemplate(messages)
    chain = prompt | chat_model
    res = chain.invoke(
        {
            "user_name": user_name,
            "query": query,
            "context": [f"\n{txt}\n" for txt in context],
        }
    )
    return str(res.content)
