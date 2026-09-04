from __future__ import annotations

import time
from typing import Dict, List, Optional, TypedDict


class AuctionState(TypedDict, total=False):
    group_chat_id: str
    ledger: Dict[str, int]                     # phone number -> token balance
    current_highest_bid: int
    current_highest_bidder: Optional[str]
    active_track_uri: Optional[str]
    active_track_name: Optional[str]
    auction_end_time: float                    # epoch seconds
    chat_logs: List[dict]                      # [{"sender", "msg", "ts"}, ...]

    auction_active: bool
    extensions_used: int

    pending_sender: Optional[str]
    pending_message: Optional[str]
    parsed_command: Optional[dict]
    outgoing_reply: Optional[str]
    validation_passed: Optional[bool]
    resolution_passed: Optional[bool]


def new_auction_state(group_chat_id: str) -> AuctionState:
    """Fresh state for a group chat that has never bid before."""
    return AuctionState(
        group_chat_id=group_chat_id,
        ledger={},
        current_highest_bid=0,
        current_highest_bidder=None,
        active_track_uri=None,
        active_track_name=None,
        auction_end_time=0.0,
        chat_logs=[],
        auction_active=False,
        extensions_used=0,
    )


def log_event(state: AuctionState, sender: str, msg: str) -> List[dict]:
    """Returns a new chat_logs list with one entry appended (state is
    treated as immutable input; nodes return the new value for LangGraph
    to merge)."""
    logs = list(state.get("chat_logs", []))
    logs.append({"sender": sender, "msg": msg, "ts": time.time()})
    return logs


def ensure_balance(ledger: Dict[str, int], sender: str, starting_balance: int) -> Dict[str, int]:
    """Returns a new ledger dict, adding `sender` at the starting balance
    if they're not already in it. New dict, not a mutation, so LangGraph's
    state merge sees a real change."""
    if sender in ledger:
        return ledger
    new_ledger = dict(ledger)
    new_ledger[sender] = starting_balance
    return new_ledger
