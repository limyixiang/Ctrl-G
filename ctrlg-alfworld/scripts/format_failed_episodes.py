"""Render failed evaluation episodes as a readable Markdown report."""

import argparse
import json
import re
import sys
from pathlib import Path


def indented(value: object) -> str:
    """Keep source text readable without letting it become Markdown markup."""
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2)
    return "\n".join("    " + line for line in (value.splitlines() or [""]))


def load_failures(path: Path) -> tuple[int, list[tuple[int, dict]]]:
    failures = []
    count = 0
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(episode, dict) or not isinstance(episode.get("success"), bool):
                raise ValueError(f"{path}:{line_number}: expected an episode with boolean success")
            count += 1
            if episode["success"] is False:
                if not isinstance(episode.get("steps"), list):
                    raise ValueError(f"{path}:{line_number}: failed episode has no steps list")
                failures.append((count, episode))
    return count, failures


def objective_from_gamefile(gamefile: str) -> str | None:
    """Recover the task type and object classes encoded in ALFWorld paths."""
    task_types = {
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_two_obj_and_place",
        "look_at_obj_in_light",
    }
    parts = gamefile.replace("\\", "/").split("/")
    task_parts = next(
        (name.split("-") for name in parts if name.split("-")[0] in task_types),
        None,
    )
    if task_parts is None or len(task_parts) < 4:
        return None
    task_type, object_name, _, destination = task_parts[:4]

    def words(name: str) -> str:
        familiar = {"CounterTop": "countertop", "KeyChain": "keychain"}
        return familiar.get(name, re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower())

    def article(name: str) -> str:
        return "an" if name[0].lower() in "aeiou" else "a"

    obj = words(object_name)
    dest = words(destination)
    preposition = "on" if dest in {"countertop", "desk", "shelf", "table", "bed", "sofa"} else "in"
    if task_type == "pick_two_obj_and_place":
        plural = obj[:-2] + "ves" if obj.endswith("fe") else obj + "s"
        return f"Place two {plural} {preposition} {article(dest)} {dest}."
    if task_type == "look_at_obj_in_light":
        return f"Examine {article(obj)} {obj} under {article(dest)} {dest}."
    operation = {
        "pick_and_place_simple": None,
        "pick_clean_then_place_in_recep": "Clean",
        "pick_heat_then_place_in_recep": "Heat",
        "pick_cool_then_place_in_recep": "Cool",
    }[task_type]
    target = f"{article(obj)} {obj}"
    placement = f"place it {preposition} {article(dest)} {dest}"
    return f"{operation} {target} and {placement}." if operation else f"Pick up {target} and {placement}."


def objective(episode: dict) -> tuple[str, str]:
    saved = episode.get("task_description")
    if isinstance(saved, str) and saved.strip():
        return "Task description", saved.strip()
    inferred = objective_from_gamefile(str(episode.get("gamefile", "")))
    if inferred:
        return "Objective inferred from gamefile", inferred
    return "Task description", "Unavailable in this evaluation record."


def render(total: int, failures: list[tuple[int, dict]], full: bool) -> str:
    lines = [
        "# Failed evaluation episodes",
        "",
        f"Failed: {len(failures)} of {total} episodes.",
        "",
    ]
    for episode_number, episode in failures:
        objective_label, objective_text = objective(episode)
        steps = episode["steps"]
        bad_steps = sum(step.get("action_was_admissible") is False for step in steps)
        lines.extend([
            f"## Episode {episode_number} — {episode.get('task_key', 'unknown task')}",
            "",
            f"{objective_label}:",
            "",
            indented(objective_text),
            "",
            f"Steps: {len(steps)} | Inadmissible actions: {bad_steps}",
            "",
            "Gamefile:",
            "",
            indented(episode.get("gamefile", "unknown")),
            "",
        ])
        for index, step in enumerate(steps):
            action = str(step.get("action", "<missing action>")).replace("\n", " ")
            admissible = step.get("action_was_admissible")
            marker = " **INADMISSIBLE**" if admissible is False else ""
            fields = step.get("decision_fields") or {}
            phase = fields.get("phase", "unknown") if isinstance(fields, dict) else "unknown"
            policies = step.get("activated_policies") or []
            lines.extend([
                f"### Step {step.get('step', index)}: {action}{marker}",
                "",
                f"- Phase: {phase}",
                f"- Admissible: {'yes' if admissible is True else 'no' if admissible is False else 'unknown'}",
                f"- Active policies: {', '.join(policies) if policies else 'none'}",
                f"- Policy fallback: {step.get('policy_fallback_reason') or 'none'}",
                "",
                "Observation:",
                "",
                indented(step.get("observation", "")),
                "",
            ])
            if admissible is False or full:
                lines.extend([
                    "Decision fields:",
                    "",
                    indented(fields),
                    "",
                ])
            if full:
                for label, key in (
                    ("Thought", "thought"),
                    ("Raw decision", "decision"),
                    ("Admissible commands", "admissible_gt"),
                ):
                    lines.extend([label + ":", "", indented(step.get(key, "")), ""])
        lines.extend(["---", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="eval_decision_*.jsonl file")
    parser.add_argument("--out", type=Path, help="write Markdown here (default: stdout)")
    parser.add_argument(
        "--full",
        action="store_true",
        help="also include each step's thought, raw decision, and admissible commands",
    )
    args = parser.parse_args()

    try:
        total, failures = load_failures(args.input)
        report = render(total, failures, args.full)
        if args.out:
            if args.input.resolve() == args.out.resolve():
                raise ValueError("--out must differ from the input file")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(report, encoding="utf-8")
            print(f"Wrote {len(failures)} failed episodes to {args.out}")
        else:
            sys.stdout.write(report)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
