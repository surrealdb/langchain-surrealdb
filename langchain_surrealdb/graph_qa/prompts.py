from langchain_core.prompts import PromptTemplate

SURQL_GENERATION_TEMPLATE = """Task: Generate a SurrealDB (surql) graph query from a User Input.

You are a SurrealDB (surql) expert responsible for translating a `User Input` into a SurrealDB graph query.

You are given a `Graph Schema`. It is a JSON Object containing:
1. `nodes`: a list of available node names
2. `edges`: a list of available edge names

You may also be given a set of `surql query examples` to help you create the `SurrealDB query`. If provided, the
`surql query examples` should be used as a reference, similar to how `Graph Schema` should be used.

Things you should do:
- Think step by step
- Generate only one SELECT query

Things you should not do:
- Do not provide explanations or apologies in your responses.
- Do not generate a surql query that removes or deletes any data.

Under no circumstance should you generate a surql query that deletes any data whatsoever.

Graph Schema:
{surql_schema}

SurrealDB Query Examples:
{surql_examples}

User Input:
{user_input}

SurrealDB Query:
"""  # noqa: E501

SURQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["surql_schema", "surql_examples", "user_input"],
    template=SURQL_GENERATION_TEMPLATE,
)

"""
Error examples:
"There was a problem with the database: Parse error: Unexpected token `a strand`, expected `?`, `(` or an identifier
 --> [1:15]
  |
1 | SELECT *,->?->'graph_Symptom' FROM graph_source WHERE ?->'graph_Condition' = ...
  |               ^
"
"""  # noqa: E501

SURQL_FIX_TEMPLATE = """Task: Address the SurrealDB Query Language (surql) error message of an SurrealDB Query Language
query.

You are an SurrealDB Query Language (surql) expert responsible for correcting the provided `surql Query` based on the
provided `surql Error`. 

The `surql Error` explains why the `surql Query` could not be executed in the database.

The `surql Error` may also contain the position of the error relative to the total number of lines of the `surql Query`.
For example, '--> [1:15]' denotes that the error X occurs on line 1, column 15 of the `surql Query`.

You are given a `Graph Schema`. It is a JSON Object containing:
1. `nodes`: a list of available node names
2. `edges`: a list of available edge names

You will output the `Corrected surql Query` wrapped in 3 backticks (```). Do not include any text except the Corrected
surql Query.

Remember to think step by step, and to generate only 1 query.

Graph Schema:
{surql_schema}

surql Query:
{surql_query}

surql Error:
{surql_error}

Corrected surql Query:
"""  # noqa: E501

SURQL_FIX_PROMPT = PromptTemplate(
    input_variables=["surql_schema", "surql_query", "surql_error"],
    template=SURQL_FIX_TEMPLATE,
)

SURQL_QA_TEMPLATE = """Task: Generate a natural language `Summary` from the results of an SurrealDB Query Language
query.

You are an SurrealDB Query Language (surql) expert responsible for creating a well-written `Summary` from the
`User Input` and associated `surql Result`.

A user has executed an SurrealDB Query Language query, which has returned the surql Result in JSON format.
You are responsible for creating an `Summary` based on the surql Result.

You are given the following information:
- `Graph Schema`: contains a schema representation of the user's SurrealDB Database.
- `User Input`: the original question/request of the user, which has been translated into an surql Query.
- `surql Query`: the surql equivalent of the `User Input`, translated by another AI Model. Should you deem it to be
incorrect, suggest a different surql Query.
- `surql Result`: the JSON output returned by executing the `surql Query` within the SurrealDB Database.

Remember to think step by step.

Your `Summary` should sound like it is a response to the `User Input`.
Your `Summary` should not include any mention of the `surql Query` or the `surql Result`.

Do not explain how you came up with the `Summary` or give any introduction. Just give the summary.

Graph Schema:
{surql_schema}

User Input:
{user_input}

surql Query:
{surql_query}

surql Result:
{surql_result}
"""  # noqa: E501

SURQL_QA_PROMPT = PromptTemplate(
    input_variables=["surql_schema", "user_input", "surql_query", "surql_result"],
    template=SURQL_QA_TEMPLATE,
)
