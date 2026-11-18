"""

Example: ARC compaction engine in a minimal (LLM-free) agent loop

Demonstrates the three integration touchpoints of
:class:`onecomp.compaction.ArcEngine` with a scripted policy standing in
for the LLM, so the full flow — observe, compact, cite, recall — runs
deterministically with no API key, no GPU, and no agent framework.

Usage::

    python example/compaction/example_arc_engine.py

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuma Ichikawa

"""

import re

from onecomp.compaction import ArcEngine, ArcEngineConfig, RecallError, sanitize_messages


def fake_environment(cmd: str) -> str:
    """Stand-in for a real execution environment (shell, tool, API...)."""
    if cmd == "cat data.log":
        return "SECRET=xyz-42-abc\n" + "noise line\n" * 800  # long observation
    if cmd == "ls":
        return "data.log  main.py"
    return f"(no output for: {cmd})"


def main():
    engine = ArcEngine(ArcEngineConfig(recall_per_step=2, keep_recent=1))

    # Touchpoint 1: the recall protocol goes into the system prompt.
    messages = [
        {"role": "system", "content": "You are an agent.\n\n" + engine.system_prompt_snippet()},
        {"role": "user", "content": "Find the SECRET in data.log and report it."},
    ]

    # A scripted "LLM": explores, then recalls the citation, then answers.
    scripted_actions = ["ls", "cat data.log", "echo working...", None, None]

    for step_no, action in enumerate(scripted_actions, 1):
        engine.begin_step()

        if action is None:
            # After compaction the big observation is only a citation; the
            # "LLM" reads the relevant §id (the `cat data.log` citation)
            # from the summary and recalls it.
            summary = messages[2]["content"]
            m = re.search(r"\u00a7([0-9a-f]{8})\s+cmd='cat data.log'", summary)
            action = f"_recall \u00a7{m.group(1)}"

        messages.append({"role": "assistant", "content": f"```bash\n{action}\n```"})
        print(f"[step {step_no}] action: {action[:60]}")

        # Touchpoint 2: intercept recalls before executing.
        try:
            outcome = engine.handle_action(action, messages)
        except RecallError as exc:
            messages.append({"role": "user", "content": str(exc)})
            continue

        if outcome is not None:
            print(
                f"           recalled \u00a7{outcome.obs_id} "
                f"({len(outcome.message['content'])} chars re-injected)"
            )
            if "SECRET=" in outcome.message["content"]:
                secret = outcome.message["content"].split("SECRET=")[1].split()[0]
                print(f"           -> found SECRET: {secret}")
                break
            continue

        output = fake_environment(action)
        rec = engine.record_observation(command=action, content=output)
        messages.append(
            engine.make_observation_message(rec, rendered=f"<output>\n{output}\n</output>")
        )

        # Touchpoint 3: proactive budget enforcement (here: a small budget
        # to force compaction of the long observation; real callers derive
        # it from the model's context window).
        messages = engine.enforce_char_budget(messages, max_chars=4_000)

    print()
    print(f"observations stored : {engine.store.n_records}")
    print(f"compactions         : {len(engine.compaction_log)}")
    print(f"recalls             : {len(engine.recall_log)}")
    print(
        f"clean for LLM API   : {all(set(m) <= {'role', 'content'} for m in sanitize_messages(messages))}"
    )


if __name__ == "__main__":
    main()
