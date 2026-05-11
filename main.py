"""Entry point for the autonomous PDE research agent."""

import argparse
from pathlib import Path

from agent.llm_interface import LLMInterface
from agent.memory import Memory
from agent.orchestrator import ResearchAgent
from agent.tools import file_io, log_experiment, run_python, submit_final

DEEPSEEK_API_KEY = "sk-6397bd4d9dbd4c45b21940676618c39a"
DEEPSEEK_MODEL = "deepseek-v4-pro"


def build_tools() -> dict:
    return {
        "run_python": run_python,
        "file_io": file_io,
        "log_experiment": log_experiment,
        "submit_final": submit_final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDE research agent")
    parser.add_argument("--task", type=int, default=1, choices=[1, 2], help="Task ID (1 or 2)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-minutes", type=int, default=480)
    args = parser.parse_args()

    llm = LLMInterface(
        api_key=DEEPSEEK_API_KEY,
        model=DEEPSEEK_MODEL,
        provider="deepseek",
        reasoning_effort="max",
        thinking_enabled=True,
    )

    # 设置日志文件路径（符合比赛要求的JSON格式）
    log_file = Path("submission") / f"task{args.task}_logs.log"
    llm.log_file = log_file

    memory = Memory(max_turns=20)
    agent = ResearchAgent(
        llm=llm,
        memory=memory,
        tools=build_tools(),
        max_steps=args.max_steps,
        max_time_minutes=args.max_minutes,
    )

    agent.run(task_id=args.task)


if __name__ == "__main__":
    main()
