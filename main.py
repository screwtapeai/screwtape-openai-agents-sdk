"""OpenAI Agents SDK: a Screwtape.ai output guardrail.

pass  -> the run completes normally
retry -> the tripwire fires; we re-run with the judge's reasons (bounded)
fail  -> the tripwire fires; we stop

pip install -r requirements.txt
export OPENAI_API_KEY=...  SCREWTAPE_API_KEY=...
python main.py
"""
import asyncio

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)

from screwtape import judge

TASK = "State the crew size and launch site of the mission."
EVIDENCE = ["The mission launched on 2024-03-01 from Cape Canaveral.", "The crew size was 4."]
CRITERIA = ["crew size", "Cape Canaveral"]
MAX_ATTEMPTS = 3


@output_guardrail
async def screwtape_check(ctx: RunContextWrapper, agent: Agent, output: str) -> GuardrailFunctionOutput:
    # judge() is a blocking HTTP call; run it off the event loop.
    v = await asyncio.to_thread(judge, TASK, str(output), EVIDENCE, CRITERIA)
    return GuardrailFunctionOutput(output_info=v, tripwire_triggered=v["verdict"] != "pass")


worker = Agent(
    name="Worker",
    instructions="Answer only from the evidence you are given. Be brief.",
    model="gpt-4o-mini",
    output_guardrails=[screwtape_check],
)


async def main() -> None:
    prompt = f"Task: {TASK}\nEvidence:\n- " + "\n- ".join(EVIDENCE)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await Runner.run(worker, prompt)
            print("ANSWER:", result.final_output)
            return
        except OutputGuardrailTripwireTriggered as tripped:
            v = tripped.guardrail_result.output.output_info
            print(f"attempt {attempt}: {v['verdict']} score={v['score']} id={v['request_id']} receipt={(v.get('receipt') or {}).get('receipt_id', '-')}")
            if v["verdict"] == "fail":
                break
            prompt += "\nA reviewer rejected your last answer because:\n- " + "\n- ".join(v["reasons"])
    print("ANSWER: (rejected)")


if __name__ == "__main__":
    asyncio.run(main())
