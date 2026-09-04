"""
nodes.py - Graph node implementations for the Aux Cord Auction Bot.

Every node takes an AuctionState and returns a *partial* dict of the keys
it changed - that's the LangGraph convention that lets the framework merge
node output back into the running state without each node having to know
the whole schema.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional, Tuple

from pydantic import BaseModel, Field

from config import settings
from state import AuctionState, ensure_balance, log_event

logger = logging.getLogger("aux_cord_bot.nodes")


# ---------------------------------------------------------------------------
# 1. ingress_parser_node
# ---------------------------------------------------------------------------

class ParsedCommand(BaseModel):
    intent: str = Field(description="One of: bid, balance, veto, reset, leaderboard, status, help, tick, unknown")
    song_query: Optional[str] = Field(default=None, description="Free-text song/artist search string")
    bid_amount: Optional[int] = Field(default=None, description="Number of Vibe Tokens bid")


_BALANCE_RE = re.compile(r"^\s*!?balance\b", re.IGNORECASE)
_VETO_RE = re.compile(r"^\s*!?veto\b", re.IGNORECASE)
_RESET_RE = re.compile(r"^\s*!?(?:reset|topup)\b", re.IGNORECASE)
_LEADERBOARD_RE = re.compile(r"^\s*!?(?:leaderboard|lb|top|ranks)\b", re.IGNORECASE)
_STATUS_RE = re.compile(r"^\s*!?(?:status|nowplaying|np|clock)\b", re.IGNORECASE)
_HELP_RE = re.compile(r"^\s*!?(?:help|commands|menu)\b", re.IGNORECASE)

# Explicit slash-command style: "!bid Espresso by Sabrina Carpenter 20" or "!bid Espresso 20 tokens"
_EXPLICIT_BID_RE = re.compile(
    r"^\s*!bid\s+(?P<song>.+?)\s+(?P<tokens>\d+)\s*(?:tokens?)?\s*$", re.IGNORECASE
)

# Natural language, verb-first: "put on / queue / play / add <song> and bid <N> [tokens]"
_NL_VERB_FIRST_RE = re.compile(
    r"(?:put on|queue up|queue|play|add)\s+(?P<song>.+?)\s+and\s+bid\s+(?P<tokens>\d+)\s*(?:tokens?)?",
    re.IGNORECASE,
)

# Natural language, bid-first: "bid 30 tokens on/for Levitating by Dua Lipa"
_NL_BID_FIRST_RE = re.compile(
    r"bid\s+(?P<tokens>\d+)\s*(?:tokens?)?\s+(?:on|for)\s+(?P<song>.+)", re.IGNORECASE
)

# Loosest fallback: just find "bid <N>" and "on/for <song>" anywhere, in any order.
_LOOSE_TOKENS_RE = re.compile(r"bid\s+(?P<tokens>\d+)", re.IGNORECASE)
_LOOSE_SONG_RE = re.compile(
    r"(?:put on|queue up|queue|play|add|on|for)\s+(?P<song>[a-z0-9][^.,!]*)", re.IGNORECASE
)


def regex_parse(text: str) -> Optional[ParsedCommand]:
    """Fast, free, deterministic parse. Returns None if nothing matched,
    signalling the caller to fall back to the LLM (if enabled)."""
    if _BALANCE_RE.search(text):
        return ParsedCommand(intent="balance")
    if _VETO_RE.search(text):
        return ParsedCommand(intent="veto")
    if _RESET_RE.search(text):
        return ParsedCommand(intent="reset")
    if _LEADERBOARD_RE.search(text):
        return ParsedCommand(intent="leaderboard")
    if _STATUS_RE.search(text):
        return ParsedCommand(intent="status")
    if _HELP_RE.search(text):
        return ParsedCommand(intent="help")

    for pattern in (_EXPLICIT_BID_RE, _NL_VERB_FIRST_RE, _NL_BID_FIRST_RE):
        m = pattern.search(text)
        if m:
            return ParsedCommand(
                intent="bid",
                song_query=m.group("song").strip(strip_chars()),
                bid_amount=int(m.group("tokens")),
            )

    # Loose fallback: both pieces present somewhere, order unknown.
    tok_m = _LOOSE_TOKENS_RE.search(text)
    song_m = _LOOSE_SONG_RE.search(text)
    if tok_m and song_m:
        return ParsedCommand(
            intent="bid",
            song_query=song_m.group("song").strip(strip_chars()),
            bid_amount=int(tok_m.group("tokens")),
        )

    return None


def strip_chars() -> str:
    return " \t\n.,!?\"'"


def llm_parse(text: str) -> Optional[ParsedCommand]:
    """LLM fallback for messages the regex genuinely can't handle
    (e.g. heavy slang, typos scrambling the keywords). Only invoked when
    USE_LLM_FALLBACK is true and an API key is configured — this keeps the
    bot fully runnable, for free, on regex alone."""
    if not settings.use_llm_fallback or not settings.openai_api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.llm_model, api_key=settings.openai_api_key, temperature=0
        ).with_structured_output(ParsedCommand)
        result = llm.invoke(
            "Extract the user's intent from this party-chat message. "
            "intent must be one of: bid, balance, veto, reset, leaderboard, status, help, unknown. "
            "If intent is 'bid', extract song_query (song name, optionally with artist) "
            "and bid_amount (integer number of tokens). "
            f"Message: {text!r}"
        )
        return result  # type: ignore[return-value]
    except Exception:
        logger.exception("LLM fallback parse failed")
        return None


TICK_SENTINEL = "__TICK__"


def ingress_parser_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    text = state.get("pending_message") or ""

    # Every invocation starts here, so this is the one place we can
    # reliably clear last turn's reply. Without this, a node that has
    # nothing new to say (e.g. a tick that finds the timer hasn't expired
    # yet) would silently re-send whatever the *previous* turn's reply
    # was, because LangGraph only overwrites keys a node actually returns.
    cleared_reply = {"outgoing_reply": None}

    # Scheduler-driven timeout check, not a real chat message, route straight to the tick pathway without running it through the parser.
    if text == TICK_SENTINEL:
        return {**cleared_reply, "parsed_command": {"intent": "tick"}}

    logger.info("ingress_parser_node: sender=%s text=%r", sender, text)

    parsed = regex_parse(text)
    if parsed is None:
        parsed = llm_parse(text)
    if parsed is None:
        parsed = ParsedCommand(intent="unknown")
        logger.warning("ingress_parser_node: could not parse message from %s: %r", sender, text)

    return {**cleared_reply, "parsed_command": parsed.model_dump()}


# ---------------------------------------------------------------------------
# 2. ledger_validator_node
# ---------------------------------------------------------------------------

def ledger_validator_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    cmd = state.get("parsed_command") or {}
    bid_amount = cmd.get("bid_amount")

    ledger = ensure_balance(dict(state.get("ledger", {})), sender, settings.starting_token_balance)
    balance = ledger[sender]
    current_high = state.get("current_highest_bid", 0)
    current_bidder = state.get("current_highest_bidder")

    if bid_amount is None or bid_amount <= 0:
        msg = f"⚠️ {sender}: couldn't find a valid token amount in your bid."
        logger.warning("ledger_validator_node: invalid bid_amount from %s: %r", sender, bid_amount)
        return {
            "ledger": ledger,
            "validation_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    if bid_amount > balance:
        msg = (
            f"⚠️ {sender}: you only have {balance} Vibe Tokens left — "
            f"can't bid {bid_amount}."
        )
        logger.warning("ledger_validator_node: %s tried to bid %d with balance %d", sender, bid_amount, balance)
        return {
            "ledger": ledger,
            "validation_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    if bid_amount == current_high and current_bidder is not None:
        msg = (
            f"⚠️ {sender}: {bid_amount} tokens ties the current bid from "
            f"{current_bidder} — you need to strictly beat it, not match it."
        )
        logger.warning("ledger_validator_node: tie bid from %s at %d", sender, bid_amount)
        return {
            "ledger": ledger,
            "validation_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    if bid_amount <= current_high:
        msg = (
            f"⚠️ {sender}: {bid_amount} tokens doesn't beat the current "
            f"high bid of {current_high} from {current_bidder}."
        )
        logger.warning("ledger_validator_node: insufficient raise from %s: %d <= %d", sender, bid_amount, current_high)
        return {
            "ledger": ledger,
            "validation_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    if sender == current_bidder:
        msg = f"⚠️ {sender}: you're already the highest bidder — no need to bid against yourself."
        logger.warning("ledger_validator_node: %s tried to outbid themself", sender)
        return {
            "ledger": ledger,
            "validation_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    logger.info("ledger_validator_node: %s's bid of %d passes validation", sender, bid_amount)
    return {"ledger": ledger, "validation_passed": True}


# ---------------------------------------------------------------------------
# 3. spotify_resolver_node
# ---------------------------------------------------------------------------

def _default_spotify_search(query: str) -> list:
    """Real Spotify search via spotipy. Swapped out for a mock in tests
    via the `search_fn` param below — never hits the network in the test
    walkthrough."""
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
        )
    )
    results = sp.search(q=query, type="track", limit=5)
    tracks = results.get("tracks", {}).get("items", [])
    return [
        {
            "uri": t["uri"],
            "name": t["name"],
            "artists": ", ".join(a["name"] for a in t["artists"]),
        }
        for t in tracks
    ]


def spotify_resolver_node(state: AuctionState, search_fn=_default_spotify_search) -> dict:
    sender = state.get("pending_sender") or "unknown"
    cmd = state.get("parsed_command") or {}
    query = cmd.get("song_query") or ""

    if not query.strip():
        msg = f"⚠️ {sender}: I couldn't tell what song you meant — try `!bid <song name> <tokens>`."
        logger.warning("spotify_resolver_node: empty song query from %s", sender)
        return {
            "resolution_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    try:
        candidates = search_fn(query)
    except Exception:
        logger.exception("spotify_resolver_node: search failed for query %r", query)
        msg = f"⚠️ Spotify search hiccupped looking up '{query}' — try again in a moment."
        return {
            "resolution_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    if not candidates:
        msg = f"⚠️ {sender}: couldn't find '{query}' on Spotify — check the spelling?"
        logger.warning("spotify_resolver_node: no results for %r", query)
        return {
            "resolution_passed": False,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    top = candidates[0]
    note = ""
    if len(candidates) > 1:
        # Edge case: multiple tracks found. We take the top search hit
        # (Spotify's own relevance ranking) rather than blocking on a
        # disambiguation round-trip mid-auction, but we tell the bidder
        # what else was in the running so they can !veto and re-bid if
        # it picked the wrong one.
        alt_names = "; ".join(f"{c['name']} — {c['artists']}" for c in candidates[1:3])
        note = f" (also considered: {alt_names})"
        logger.info("spotify_resolver_node: %d candidates for %r, chose top hit", len(candidates), query)

    logger.info("spotify_resolver_node: resolved %r -> %s (%s)", query, top["uri"], top["name"])
    return {
        "resolution_passed": True,
        "active_track_uri": top["uri"],
        "active_track_name": f"{top['name']} — {top['artists']}{note}",
    }


# ---------------------------------------------------------------------------
# 4. auction_loop_node
# ---------------------------------------------------------------------------

def auction_loop_node(state: AuctionState) -> dict:
    """Called after a bid has passed validation + resolution. Applies the
    snipe-prevention protocol and records the new leader."""
    sender = state.get("pending_sender") or "unknown"
    cmd = state.get("parsed_command") or {}
    bid_amount = cmd["bid_amount"]
    now = time.time()

    auction_active = state.get("auction_active", False)
    extensions_used = state.get("extensions_used", 0)

    if not auction_active:
        # Brand new round.
        end_time = now + settings.auction_duration_seconds
        msg = (
            f"🎧 New auction started! {sender} bids {bid_amount} tokens on "
            f"{state.get('active_track_name')}. {int(settings.auction_duration_seconds)}s on the clock."
        )
        logger.info("auction_loop_node: new auction started by %s for %d", sender, bid_amount)
        return {
            "auction_active": True,
            "auction_end_time": end_time,
            "current_highest_bid": bid_amount,
            "current_highest_bidder": sender,
            "extensions_used": 0,
            "outgoing_reply": msg,
            "chat_logs": log_event(state, "system", msg),
        }

    # Existing round: apply snipe protection if we're inside the window.
    time_remaining = state.get("auction_end_time", now) - now
    new_end_time = state.get("auction_end_time", now)
    extended = False

    if time_remaining <= settings.snipe_window_seconds and extensions_used < settings.max_auction_extensions:
        new_end_time = now + settings.snipe_extension_seconds
        extensions_used += 1
        extended = True
        logger.info(
            "auction_loop_node: snipe protection triggered by %s, extending %.0fs (extension %d/%d)",
            sender, settings.snipe_extension_seconds, extensions_used, settings.max_auction_extensions,
        )
    elif time_remaining <= settings.snipe_window_seconds:
        logger.info("auction_loop_node: snipe window hit but max extensions (%d) reached", settings.max_auction_extensions)

    suffix = f" ⏱️ Snipe protection: +{int(settings.snipe_extension_seconds)}s added!" if extended else ""
    msg = (
        f"🔥 {sender} outbids with {bid_amount} tokens on "
        f"{state.get('active_track_name')}!{suffix}"
    )

    return {
        "auction_end_time": new_end_time,
        "current_highest_bid": bid_amount,
        "current_highest_bidder": sender,
        "extensions_used": extensions_used,
        "outgoing_reply": msg,
        "chat_logs": log_event(state, "system", msg),
    }


# ---------------------------------------------------------------------------
# 5. balance_node / veto_node
# ---------------------------------------------------------------------------

def balance_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    ledger = ensure_balance(dict(state.get("ledger", {})), sender, settings.starting_token_balance)
    msg = f"💰 {sender}, you have {ledger[sender]} Vibe Tokens."
    logger.info("balance_node: %s balance=%d", sender, ledger[sender])
    return {"ledger": ledger, "outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}


def veto_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    if not state.get("auction_active"):
        msg = f"⚠️ {sender}: there's no active auction to veto."
        logger.info("veto_node: no-op veto from %s, nothing active", sender)
        return {"outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}

    track_name = state.get("active_track_name")
    msg = f"🛑 {sender} vetoed the round for {track_name}. No tokens spent — bidding is reopened."
    logger.info("veto_node: %s vetoed round for %r", sender, track_name)
    return {
        "auction_active": False,
        "auction_end_time": 0.0,
        "current_highest_bid": 0,
        "current_highest_bidder": None,
        "active_track_uri": None,
        "active_track_name": None,
        "extensions_used": 0,
        "outgoing_reply": msg,
        "chat_logs": log_event(state, "system", msg),
    }


def fallback_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    msg = (
        f"🤔 {sender}: didn't catch that. Try `!bid <song> <tokens>`, "
        f"`!balance`, `!status`, `!leaderboard`, `!reset`, or `!help`."
    )
    logger.info("fallback_node: unparseable message from %s", sender)
    return {"outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}


def reset_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    ledger = dict(state.get("ledger", {}))
    new_ledger = {k: settings.starting_token_balance for k in ledger}
    if sender not in new_ledger:
        new_ledger[sender] = settings.starting_token_balance
    msg = f"🔄 {sender} reset the ledger! Everyone's balance has been restored to {settings.starting_token_balance} Vibe Tokens."
    logger.info("reset_node: ledger reset by %s", sender)
    return {"ledger": new_ledger, "outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}


def leaderboard_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    ledger = ensure_balance(dict(state.get("ledger", {})), sender, settings.starting_token_balance)
    sorted_ranks = sorted(ledger.items(), key=lambda x: x[1], reverse=True)
    medals = ["🥇", "🥈", "🥉"]
    lines = ["🏆 Vibe Token Leaderboard:"]
    for idx, (user, tokens) in enumerate(sorted_ranks[:5]):
        icon = medals[idx] if idx < len(medals) else "🔹"
        lines.append(f"{icon} {user}: {tokens} tokens")
    msg = "\n".join(lines)
    logger.info("leaderboard_node: fetched leaderboard for %s", sender)
    return {"ledger": ledger, "outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}


def status_node(state: AuctionState) -> dict:
    sender = state.get("pending_sender") or "unknown"
    active = state.get("auction_active", False)
    if not active:
        msg = "💤 No active auction right now. Start one with `!bid <song> <tokens>`!"
    else:
        now = time.time()
        remaining = max(0.0, state.get("auction_end_time", now) - now)
        track = state.get("active_track_name", "Unknown Track")
        bidder = state.get("current_highest_bidder", "Unknown")
        bid = state.get("current_highest_bid", 0)
        msg = (
            f"🎧 Active Auction Status:\n"
            f"• Song: {track}\n"
            f"• High Bid: {bid} tokens by {bidder}\n"
            f"• Time Remaining: {remaining:.1f}s"
        )
    logger.info("status_node: status requested by %s", sender)
    return {"outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}


def help_node(state: AuctionState) -> dict:
    msg = (
        "📜 Aux Cord Auction Bot Commands:\n\n"
        "• !bid <song> <tokens> — Start or raise a bid on a track\n"
        "• !balance — View your remaining Vibe Tokens\n"
        "• !status — View active auction details & countdown\n"
        "• !leaderboard — View group token rankings\n"
        "• !veto — Cancel active round (refunds tokens)\n"
        "• !reset — Restore everyone's balance to starting tokens\n"
        "• !help — Display this command menu"
    )
    return {"outgoing_reply": msg, "chat_logs": log_event(state, "system", msg)}


# ---------------------------------------------------------------------------
# 6. tick_check_node — checks whether time has run out (invoked by the background scheduler)
# ---------------------------------------------------------------------------

def tick_check_node(state: AuctionState) -> dict:
    if not state.get("auction_active"):
        return {"validation_passed": False}
    now = time.time()
    expired = now >= state.get("auction_end_time", 0)
    logger.debug(
        "tick_check_node: group=%s expired=%s remaining=%.1fs",
        state.get("group_chat_id"), expired, state.get("auction_end_time", 0) - now,
    )
    return {"validation_passed": expired}  # reused as the routing flag for this pathway


# ---------------------------------------------------------------------------
# 7. egress_execution_node
# ---------------------------------------------------------------------------

def _default_add_to_queue(track_uri: str) -> None:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth

    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=settings.spotify_client_id,
            client_secret=settings.spotify_client_secret,
            redirect_uri=settings.spotify_redirect_uri,
            scope=settings.spotify_scope,
            cache_path=settings.spotify_token_cache_path,
        )
    )
    devices_resp = sp.devices()
    devices = devices_resp.get("devices", []) if isinstance(devices_resp, dict) else []
    if devices:
        active_device = next((d for d in devices if d.get("is_active")), devices[0])
        sp.add_to_queue(track_uri, device_id=active_device["id"])
    else:
        sp.add_to_queue(track_uri)


def egress_execution_node(state: AuctionState, queue_fn=_default_add_to_queue) -> dict:
    winner = state.get("current_highest_bidder")
    bid = state.get("current_highest_bid", 0)
    track_uri = state.get("active_track_uri")
    track_name = state.get("active_track_name")
    group = state.get("group_chat_id")

    if not winner or not track_uri:
        logger.warning("egress_execution_node: fired with no winner/track for group %s — resetting only", group)
        reply = "⚠️ Auction closed with no valid winner — reopening the floor." if state.get("auction_active") else ""
        return _reset_round(state, reply)

    ledger = dict(state.get("ledger", {}))
    if winner not in ledger or ledger[winner] < bid:
        logger.error("egress_execution_node: winner %s has insufficient balance at execution time!", winner)
        msg = f"⚠️ Couldn't finalize {winner}'s winning bid — balance changed mid-auction. Round voided."
        return _reset_round(state, msg)

    try:
        queue_fn(track_uri)
    except Exception as exc:
        logger.exception("egress_execution_node: Spotify add_to_queue failed for %s", track_uri)
        err_str = str(exc)
        if "No active device" in err_str or "device" in err_str.lower():
            msg = (
                f"⚠️ {winner} won with '{track_name}', but Spotify rejected the queue: No active Spotify player found! "
                "Make sure Spotify is open on your computer/phone and start playing a song. (Tokens refunded)."
            )
        else:
            msg = f"⚠️ Won the bid but Spotify rejected the queue add for {track_name}. Tokens refunded."
        return _reset_round(state, msg)

    ledger[winner] = ledger[winner] - bid
    msg = f"🏆 {winner} wins the aux with {track_name} for {bid} tokens! Now playing next up."
    logger.info("egress_execution_node: %s won %r for %d tokens, queued to Spotify", winner, track_name, bid)

    reset = _reset_round(state, msg)
    reset["ledger"] = ledger
    return reset


def _reset_round(state: AuctionState, msg: str) -> dict:
    res = {
        "auction_active": False,
        "auction_end_time": 0.0,
        "current_highest_bid": 0,
        "current_highest_bidder": None,
        "active_track_uri": None,
        "active_track_name": None,
        "extensions_used": 0,
        "validation_passed": False,
    }
    if msg:
        res["outgoing_reply"] = msg
        res["chat_logs"] = log_event(state, "system", msg)
    else:
        res["outgoing_reply"] = None
    return res
