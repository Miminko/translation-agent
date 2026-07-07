from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Set, TypedDict

from langgraph.graph import END, StateGraph

from agents import critic, repair
from config import settings
from state.models import Job, Segment


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class RefinementState(TypedDict):
    segments: List[Segment]
    iteration: int
    repair_ids: Set[int]
    done: bool


@dataclass
class RefinementSummary:
    iterations: int = 0
    total_flagged: int = 0
    total_repaired: int = 0
    log: list[dict] = field(default_factory=list)


def _route_after_critique(state: RefinementState) -> str:
    if not state["repair_ids"]:
        return "finish"
    if state["iteration"] >= settings.refinement_max_iterations:
        return "finish"
    return "repair"


def _route_after_repair(state: RefinementState) -> str:
    if state["iteration"] >= settings.refinement_max_iterations:
        return "finish"
    return "critique"


def refine_segments(
    segments: List[Segment],
    job: Job,
    *,
    enabled: Optional[bool] = None,
    verbose: bool = False,
    on_progress: Optional[Callable[[List[Segment], int, int], None]] = None,
    log_path: Optional[Path] = None,
) -> tuple[List[Segment], RefinementSummary]:
    """Translate → critique → repair loop via LangGraph."""
    use_refinement = settings.refinement_enabled if enabled is None else enabled
    if not segments or not use_refinement:
        return segments, RefinementSummary()

    summary = RefinementSummary()
    only_ids: Optional[Set[int]] = None

    def critique_step(state: RefinementState) -> RefinementState:
        nonlocal only_ids
        if verbose:
            phase = "re-critique" if state["iteration"] > 0 else "critique"
            _log(f"  refinement: {phase} (iteration {state['iteration'] + 1})")
        updated, repair_ids = critic.critique_segments(
            state["segments"],
            job,
            only_ids=only_ids,
            verbose=verbose,
            on_progress=on_progress,
        )
        summary.total_flagged += len(repair_ids)
        summary.log.append(
            {
                "iteration": state["iteration"] + 1,
                "phase": "critique",
                "flagged": len(repair_ids),
            }
        )
        only_ids = None
        return {
            "segments": updated,
            "iteration": state["iteration"],
            "repair_ids": repair_ids,
            "done": False,
        }

    def repair_step(state: RefinementState) -> RefinementState:
        if verbose:
            _log(
                f"  refinement: repair {len(state['repair_ids'])} segments "
                f"(iteration {state['iteration'] + 1})"
            )
        updated = repair.repair_segments(
            state["segments"],
            job,
            state["repair_ids"],
            verbose=verbose,
            on_progress=on_progress,
        )
        summary.total_repaired += len(state["repair_ids"])
        summary.log.append(
            {
                "iteration": state["iteration"] + 1,
                "phase": "repair",
                "repaired": len(state["repair_ids"]),
            }
        )
        nonlocal only_ids
        only_ids = set(state["repair_ids"])
        return {
            "segments": updated,
            "iteration": state["iteration"] + 1,
            "repair_ids": set(),
            "done": False,
        }

    graph = StateGraph(RefinementState)
    graph.add_node("critique", critique_step)
    graph.add_node("repair", repair_step)
    graph.set_entry_point("critique")
    graph.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"repair": "repair", "finish": END},
    )
    graph.add_conditional_edges(
        "repair",
        _route_after_repair,
        {"critique": "critique", "finish": END},
    )

    compiled = graph.compile()
    initial: RefinementState = {
        "segments": segments,
        "iteration": 0,
        "repair_ids": set(),
        "done": False,
    }
    final = compiled.invoke(initial)
    summary.iterations = final["iteration"]

    if log_path is not None:
        log_path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "max_iterations": settings.refinement_max_iterations,
                    "confidence_threshold": settings.refinement_confidence_threshold,
                    "iterations": summary.iterations,
                    "total_flagged": summary.total_flagged,
                    "total_repaired": summary.total_repaired,
                    "events": summary.log,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    return final["segments"], summary
