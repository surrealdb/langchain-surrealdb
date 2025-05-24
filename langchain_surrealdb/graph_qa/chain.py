import re
from typing import Any, Dict, List, Optional, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import Field

from langchain_surrealdb.graph_qa.prompts import (
    SURQL_FIX_PROMPT,
    SURQL_GENERATION_PROMPT,
    SURQL_QA_PROMPT,
)
from langchain_surrealdb.surrealdb_graph import SurrealDBGraph

INTERMEDIATE_STEPS_KEY = "intermediate_steps"

SURQL_EXAMPLES = """
SELECT *,->?->? FROM graph_source
"""


def extract_surql(text: str) -> str:
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"
    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0] if matches else text


# TODO: implement
class SurrealDBGraphQAChain(Chain):
    graph: SurrealDBGraph = Field(exclude=True)
    qa_chain: LLMChain
    surql_generation_chain: LLMChain
    surql_fix_chain: LLMChain
    top_k: int = 10
    return_intermediate_steps: bool = False
    skip_qa_prompt: bool = False
    _input_key: str = "query"
    _output_key: str = "result"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)

    @property
    def input_keys(self) -> list[str]:
        """Keys expected to be in the chain input."""
        return [self._input_key]

    @property
    def output_keys(self) -> list[str]:
        """Keys expected to be in the chain output."""
        return [self._output_key]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        graph: SurrealDBGraph,
        qa_prompt: BasePromptTemplate = SURQL_QA_PROMPT,
        surql_generation_prompt: BasePromptTemplate = SURQL_GENERATION_PROMPT,
        surql_fix_prompt: BasePromptTemplate = SURQL_FIX_PROMPT,
        **kwargs: Any,
    ) -> "SurrealDBGraphQAChain":
        """Initialize from LLM."""
        qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        surql_generation_chain = LLMChain(llm=llm, prompt=surql_generation_prompt)
        surql_fix_chain = LLMChain(llm=llm, prompt=surql_fix_prompt)

        return cls(
            graph=graph,
            qa_chain=qa_chain,
            surql_generation_chain=surql_generation_chain,
            surql_fix_chain=surql_fix_chain,
            **kwargs,
        )

    def _call(
        self,
        inputs: dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        question = inputs[self._input_key]
        graph_schema = self.graph.get_schema
        args = {
            "user_input": question,
            "surrealdb_schema": graph_schema,
            "surql_examples": SURQL_EXAMPLES,
        }
        args.update(inputs)

        intermediate_steps: List = []

        result = self.surql_generation_chain.invoke(args, callbacks=callbacks)
        generated_surql = extract_surql(result["text"])

        # Correct Cypher query if enabled
        # if self.cypher_query_corrector:
        #     generated_cypher = self.cypher_query_corrector(generated_cypher)

        _run_manager.on_text("Generated surql:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_surql, color="green", end="\n", verbose=self.verbose
        )

        intermediate_steps.append({"query": generated_surql})

        # Retrieve and limit the number of results
        # Generated Cypher be null if query corrector identifies invalid schema
        if generated_surql:
            context = self.graph.query(generated_surql)[: self.top_k]
        else:
            context = []

        final_result: Union[List[Dict[str, Any]], str]
        if self.skip_qa_prompt:
            final_result = context
        else:
            _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
            _run_manager.on_text(
                str(context), color="green", end="\n", verbose=self.verbose
            )
            final_result = self.qa_chain.invoke(
                {
                    "user_input": question,
                    "surrealdb_schema": graph_schema,
                    "surql_result": context,
                    "surql_query": generated_surql,
                },
                callbacks=callbacks,
            )

        chain_result: Dict[str, Any] = {self._output_key: final_result}
        if self.return_intermediate_steps:
            chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps

        return chain_result
