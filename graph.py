"""
graph.py — Wires the AuctionState nodes into a compiled LangGraph app.

Topology:

    ingress_parser_node
        |
        +--(intent=bid)---------> ledger_validator_node
        |                              |
        |                              +--(pass)--> spotify_resolver_node
        |                              |                 |
        |                              |                 +--(pass)--> auction_loop_node --> END
        |                              |                 +--(fail)--> END
        |                              +--(fail)--> END
        |
        +--(intent=balance)-----> balance_node --> END
        +--(intent=veto)--------> veto_node --> END
        +--(intent=reset)-------> reset_node --> END
        +--(intent=leaderboard)-> leaderboard_node --> END
        +--(intent=status)------> status_node --> END
        +--(intent=help)--------> help_node --> END
        +--(intent=tick)--------> tick_check_node
        |                              +--(expired)--> egress_execution_node --> END
        |                              +--(not expired)--> END
        +--(intent=unknown)-----> fallback_node --> END

Two distinct call patterns hit this same graph:
  1. A real incoming Telegram message -> intent is bid/balance/veto/reset/etc.
  2. A scheduler "tick" (see telegram_app.py's background task) -> intent="tick",
     used purely to ask "has this group's timer run out yet?" without
     blocking the event loop on a real-time sleep inside the graph.

Persistence: uses LangGraph's MemorySaver checkpointer keyed by
thread_id=group_chat_id, so state.py's AuctionState survives across the
many separate `invoke()` calls that make up one auction (one call per
incoming message, plus periodic ticks). Swap MemorySaver for a durable
checkpointer (Postgres/SQLite) for production deployments.
"""
from __future__ import annotations

import functools

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes import (
    TICK_SENTINEL,
    auction_loop_node,
    balance_node,
    egress_execution_node,
    fallback_node,
    help_node,
    ingress_parser_node,
    leaderboard_node,
    ledger_validator_node,
    reset_node,
    spotify_resolver_node,
    status_node,
    tick_check_node,
    veto_node,
)
from state import AuctionState


def _route_after_ingress(state: AuctionState) -> str:
    cmd = state.get("parsed_command") or {}
    return cmd.get("intent", "unknown")


def _route_after_validation(state: AuctionState) -> str:
    return "pass" if state.get("validation_passed") else "fail"


def _route_after_resolution(state: AuctionState) -> str:
    return "pass" if state.get("resolution_passed") else "fail"


def _route_after_tick(state: AuctionState) -> str:
    return "expired" if state.get("validation_passed") else "not_expired"


def build_graph(checkpointer=None, spotify_search_fn=None, spotify_queue_fn=None):
    """`spotify_search_fn` / `spotify_queue_fn` let callers (tests, or a
    real deployment wanting a different Spotify client setup) inject
    their own implementations instead of the real network-calling
    defaults in nodes.py — this is how the mock walkthrough test runs
    the entire graph with zero network access."""
    workflow = StateGraph(AuctionState)

    resolver = (
        functools.partial(spotify_resolver_node, search_fn=spotify_search_fn)
        if spotify_search_fn is not None
        else spotify_resolver_node
    )
    egress = (
        functools.partial(egress_execution_node, queue_fn=spotify_queue_fn)
        if spotify_queue_fn is not None
        else egress_execution_node
    )

    workflow.add_node("ingress_parser_node", ingress_parser_node)
    workflow.add_node("ledger_validator_node", ledger_validator_node)
    workflow.add_node("spotify_resolver_node", resolver)
    workflow.add_node("auction_loop_node", auction_loop_node)
    workflow.add_node("balance_node", balance_node)
    workflow.add_node("veto_node", veto_node)
    workflow.add_node("reset_node", reset_node)
    workflow.add_node("leaderboard_node", leaderboard_node)
    workflow.add_node("status_node", status_node)
    workflow.add_node("help_node", help_node)
    workflow.add_node("fallback_node", fallback_node)
    workflow.add_node("tick_check_node", tick_check_node)
    workflow.add_node("egress_execution_node", egress)

    workflow.set_entry_point("ingress_parser_node")

    workflow.add_conditional_edges(
        "ingress_parser_node",
        _route_after_ingress,
        {
            "bid": "ledger_validator_node",
            "balance": "balance_node",
            "veto": "veto_node",
            "reset": "reset_node",
            "leaderboard": "leaderboard_node",
            "status": "status_node",
            "help": "help_node",
            "tick": "tick_check_node",
            "unknown": "fallback_node",
        },
    )

    workflow.add_conditional_edges(
        "ledger_validator_node",
        _route_after_validation,
        {"pass": "spotify_resolver_node", "fail": END},
    )

    workflow.add_conditional_edges(
        "spotify_resolver_node",
        _route_after_resolution,
        {"pass": "auction_loop_node", "fail": END},
    )

    workflow.add_conditional_edges(
        "tick_check_node",
        _route_after_tick,
        {"expired": "egress_execution_node", "not_expired": END},
    )

    workflow.add_edge("auction_loop_node", END)
    workflow.add_edge("balance_node", END)
    workflow.add_edge("veto_node", END)
    workflow.add_edge("reset_node", END)
    workflow.add_edge("leaderboard_node", END)
    workflow.add_edge("status_node", END)
    workflow.add_edge("help_node", END)
    workflow.add_edge("fallback_node", END)
    workflow.add_edge("egress_execution_node", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())


# Module-level singleton for app.py to import. Built with its own
# MemorySaver instance so all webhook calls and scheduler ticks share
# one checkpointer (and therefore one view of state per group_chat_id).
compiled_app = build_graph()


def invoke_for_group(group_chat_id: str, sender: str, message: str, graph_app=None) -> AuctionState:
    """Convenience wrapper: run one message through the graph for a given
    group, returning the resulting full state. `graph_app` defaults to the
    module-level singleton but can be overridden (e.g. in tests, with a
    graph built via build_graph(spotify_search_fn=mock, ...))."""
    graph_app = graph_app or compiled_app
    config = {"configurable": {"thread_id": group_chat_id}}
    result = graph_app.invoke(
        {
            "group_chat_id": group_chat_id,
            "pending_sender": sender,
            "pending_message": message,
        },
        config=config,
    )
    return result


def invoke_tick(group_chat_id: str, graph_app=None) -> AuctionState:
    """Scheduler-driven check for whether this group's auction has timed out."""
    graph_app = graph_app or compiled_app
    config = {"configurable": {"thread_id": group_chat_id}}
    result = graph_app.invoke(
        {
            "group_chat_id": group_chat_id,
            "pending_sender": "scheduler",
            "pending_message": TICK_SENTINEL,
        },
        config=config,
    )
    return result
