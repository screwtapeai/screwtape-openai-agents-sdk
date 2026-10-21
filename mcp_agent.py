"""OpenAI Agents SDK: give an agent Screwtape's MCP tools (filtered to judge and consensus).

pip install -r requirements.txt
export OPENAI_API_KEY=...  SCREWTAPE_API_KEY=...
python mcp_agent.py
"""
import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp


async def main() -> None:
    async with MCPServerStreamableHttp(
        name="screwtape",
        params={
            "url": "https://screwtape.ai/mcp",
            "headers": {"Authorization": "Bearer " + os.environ["SCREWTAPE_API_KEY"]},
        },
        cache_tools_list=True,
        tool_filter={"allowed_tool_names": ["judge", "consensus"]},
    ) as screwtape:
        reviewer = Agent(
            name="Reviewer",
            instructions=(
                "Before giving a final answer, call the judge tool with the task, your draft as agent_output, "
                "and the evidence. Only answer if the verdict is pass; on retry, revise using the reasons."
            ),
            model="gpt-4o-mini",
            mcp_servers=[screwtape],
        )
        result = await Runner.run(
            reviewer,
            "Task: State the crew size. Evidence: 'The crew size was 4.' Draft: 'The crew size was 5.'",
        )
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
