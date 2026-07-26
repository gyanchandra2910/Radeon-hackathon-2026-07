"""Async multi-agent code review and optimization using a local vLLM endpoint.

The agents use CrewAI, but every LLM call is routed to an OpenAI-compatible
vLLM server running locally on AMD Radeon GPU / ROCm infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

import httpx
from crewai import Agent, LLM
from dotenv import load_dotenv


DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

load_dotenv()


SAMPLE_CODE = dedent(
    """
    def find_duplicate_pairs(values):
        pairs = []
        seen = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i] == values[j] and values[i] not in seen:
                    pairs.append((values[i], i, j))
            seen.append(values[i])
        return pairs


    data = [3, 9, 3, 4, 9, 9, 1]
    print(find_duplicate_pairs(data))
    """
).strip()


@dataclass(frozen=True)
class OptimizerConfig:
    """Runtime settings for the local OpenAI-compatible endpoint."""

    model: str
    base_url: str
    api_key: str
    temperature: float
    timeout_seconds: int


def load_config(model_override: str | None = None, base_url_override: str | None = None) -> OptimizerConfig:
    return OptimizerConfig(
        model=model_override or os.getenv("LOCAL_VLLM_MODEL", DEFAULT_MODEL),
        base_url=(base_url_override or os.getenv("LOCAL_VLLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/"),
        api_key=os.getenv("LOCAL_VLLM_API_KEY", "EMPTY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.15")),
        timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "180")),
    )


def build_llm(config: OptimizerConfig) -> LLM:
    """Create a CrewAI LLM client pinned to the local vLLM OpenAI API."""

    return LLM(
        model=f"openai/{config.model}",
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )


def build_agents(llm: LLM) -> tuple[Agent, Agent, Agent]:
    logic_reviewer = Agent(
        role="Logic Reviewer",
        goal="Find correctness bugs, hidden edge cases, and unsafe assumptions in submitted Python or C++ code.",
        backstory=(
            "You are a senior software correctness reviewer. You focus on concrete failure modes, "
            "minimal counterexamples, data-structure invariants, and behavioral regressions."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )

    performance_reviewer = Agent(
        role="Performance Reviewer",
        goal="Identify latency bottlenecks and practical low-level improvements without changing required behavior.",
        backstory=(
            "You are an AMD ROCm performance engineer who reviews hot paths, allocations, loop structure, "
            "branching, cache behavior, and algorithmic complexity for production code."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
    )

    code_optimizer = Agent(
        role="Code Optimizer",
        goal="Rewrite code to be correct, readable, and low latency while preserving the public behavior.",
        backstory=(
            "You are a pragmatic optimization engineer. You merge review findings, keep the rewrite compact, "
            "and explain only the changes that materially affect correctness or latency."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )

    return logic_reviewer, performance_reviewer, code_optimizer


def result_text(result: Any) -> str:
    return str(getattr(result, "raw", result)).strip()


def fenced_code(code: str, language: str) -> str:
    return f"```{language}\n{code.strip()}\n```"


async def assert_vllm_ready(config: OptimizerConfig) -> None:
    models_url = f"{config.base_url}/models"
    headers = {"Authorization": f"Bearer {config.api_key}"}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(models_url, headers=headers)
        response.raise_for_status()


async def run_async_code_optimizer(code: str, language: str, config: OptimizerConfig) -> dict[str, str]:
    llm = build_llm(config)
    logic_reviewer, performance_reviewer, code_optimizer = build_agents(llm)

    code_block = fenced_code(code, language)

    logic_prompt = dedent(
        f"""
        Review this {language} code for correctness only.

        Return exactly these sections:
        1. Bugs or edge cases
        2. Minimal failing examples, if any
        3. Behavior that must be preserved

        Be specific and avoid generic advice.

        {code_block}
        """
    ).strip()

    performance_prompt = dedent(
        f"""
        Review this {language} code for low-latency execution.

        Return exactly these sections:
        1. Complexity and hot path
        2. Avoidable allocations or repeated work
        3. Recommended rewrite strategy

        Prefer practical changes that preserve readability.

        {code_block}
        """
    ).strip()

    logic_task, performance_task = await asyncio.gather(
        logic_reviewer.kickoff_async(logic_prompt),
        performance_reviewer.kickoff_async(performance_prompt),
    )

    logic_review = result_text(logic_task)
    performance_review = result_text(performance_task)

    optimizer_prompt = dedent(
        f"""
        Rewrite the original {language} code using the two reviews.

        Requirements:
        - Preserve intended behavior unless a review proves the behavior is a bug.
        - Improve asymptotic complexity or constant factors where possible.
        - Return a complete replacement program or function.
        - Include a short rationale after the code.
        - Do not mention public cloud APIs; this system runs on local vLLM.

        Original code:
        {code_block}

        Logic Reviewer findings:
        {logic_review}

        Performance Reviewer findings:
        {performance_review}
        """
    ).strip()

    optimized = await code_optimizer.kickoff_async(optimizer_prompt)

    return {
        "logic_review": logic_review,
        "performance_review": performance_review,
        "optimized_code": result_text(optimized),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an async local multi-agent code optimizer.")
    parser.add_argument("--file", type=Path, help="Python or C++ file to review. Uses a sample snippet when omitted.")
    parser.add_argument("--language", default="python", choices=["python", "cpp"], help="Input code language.")
    parser.add_argument("--model", help=f"Model name served by vLLM. Default: {DEFAULT_MODEL}")
    parser.add_argument("--base-url", help=f"OpenAI-compatible vLLM base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--skip-health-check", action="store_true", help="Skip the /v1/models readiness check.")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    config = load_config(model_override=args.model, base_url_override=args.base_url)

    code = SAMPLE_CODE
    if args.file:
        code = args.file.read_text(encoding="utf-8")
        if args.language == "python" and args.file.suffix in {".cc", ".cpp", ".cxx", ".hpp", ".h"}:
            args.language = "cpp"

    print(f"Using local vLLM endpoint: {config.base_url}")
    print(f"Using model: {config.model}")

    if not args.skip_health_check:
        await assert_vllm_ready(config)

    result = await run_async_code_optimizer(code=code, language=args.language, config=config)

    print("\n=== Logic Review ===\n")
    print(result["logic_review"])
    print("\n=== Performance Review ===\n")
    print(result["performance_review"])
    print("\n=== Optimized Output ===\n")
    print(result["optimized_code"])


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
