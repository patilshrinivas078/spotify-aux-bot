"""
Simulation of a full auction round between two users,
with Spotify search/queue fully mocked out (no network access, no API
keys required). Run with: python test_walkthrough.py

Sequence simulated:
  1. Alice checks her balance.
  2. Bob sends a messy natural-language bid: "hey bot put on espresso by
     sabrina and bid 20 tokens" — this starts a new auction.
  3. Alice outbids with an explicit "!bid Espresso 35" — same song, higher
     bid; validator must allow beating the previous bidder even on the
     same track.
  4. Bob tries to tie Alice's bid (should be rejected).
  5. Bob raises with a slash-command bid in the last few seconds of the
     window — snipe protection should extend the timer.
  6. The scheduler ticks past the (shortened, test-only) auction window.
  7. egress_execution_node fires: Alice wins, tokens deducted, track
     pushed to the (mocked) Spotify queue.
  8. Final ledger and queue state are printed for inspection.
"""
from __future__ import annotations

import time

import config as config_module
from graph import build_graph, invoke_for_group, invoke_tick

GROUP = "party-group-guid-001"


def run():
    # --- Test-only tuning: short auction window so the walkthrough
    # doesn't take 60 real seconds, and a short snipe window so we can
    # trivially land a bid inside it. ---
    config_module.settings.auction_duration_seconds = 3
    config_module.settings.snipe_window_seconds = 2
    config_module.settings.snipe_extension_seconds = 3
    config_module.settings.starting_token_balance = 100
    config_module.settings.use_llm_fallback = False  # regex only, no API key needed

    # --- Mocked Spotify layer: no network, deterministic results. ---
    fake_catalog = {
        "espresso by sabrina": [
            {"uri": "spotify:track:mock_espresso_1", "name": "Espresso", "artists": "Sabrina Carpenter"},
        ],
        "espresso": [
            {"uri": "spotify:track:mock_espresso_1", "name": "Espresso", "artists": "Sabrina Carpenter"},
            {"uri": "spotify:track:mock_espresso_2", "name": "Espresso Macchiato", "artists": "Tommy Cash"},
        ],
    }
    queued_tracks = []  # records every add_to_queue call

    def mock_search(query: str):
        key = query.strip().lower()
        if key in fake_catalog:
            return fake_catalog[key]
        # loose contains-match fallback, like a real search engine would do
        for k, v in fake_catalog.items():
            if key in k or k in key:
                return v
        return []

    def mock_queue(track_uri: str):
        queued_tracks.append(track_uri)

    test_graph = build_graph(spotify_search_fn=mock_search, spotify_queue_fn=mock_queue)

    def step(label, sender, message=None, tick=False):
        print(f"\n--- {label} ---")
        if tick:
            result = invoke_tick(GROUP, graph_app=test_graph)
        else:
            print(f"[{sender}]: {message}")
            result = invoke_for_group(GROUP, sender, message, graph_app=test_graph)
        reply = result.get("outgoing_reply")
        if reply:
            print(f"[BOT]: {reply}")
        print(
            f"    state: highest_bid={result.get('current_highest_bid')} "
            f"bidder={result.get('current_highest_bidder')} "
            f"active={result.get('auction_active')} "
            f"ends_in={round(result.get('auction_end_time', 0) - time.time(), 1)}s"
        )
        return result

    # 1. Balance check
    step("Alice checks balance", "+1-555-0100", "!balance")

    # 2. Bob's messy natural-language bid opens the auction
    step(
        "Bob opens the auction with a messy message",
        "+1-555-0200",
        "hey bot put on espresso by sabrina and bid 20 tokens",
    )

    # 3. Alice outbids explicitly
    step("Alice outbids with explicit slash command", "+1-555-0100", "!bid Espresso 35")

    # 4. Bob tries to tie — should be rejected
    step("Bob tries to tie Alice's bid (should be rejected)", "+1-555-0200", "!bid Espresso 35")

    # 5. Wait until we're inside the snipe window, then Bob raises again
    print("\n(sleeping so we land inside the snipe window...)")
    time.sleep(1.5)  # duration=3s, snipe_window=2s -> ~1.5s remaining is inside the window
    step("Bob snipes with a late raise (should extend the timer)", "+1-555-0200", "!bid Espresso 50")

    # 6. Let the (now-extended) timer actually run out, ticking periodically
    print("\n(sleeping past the extended auction window...)")
    for _ in range(6):
        time.sleep(1)
        result = step("Scheduler tick", "scheduler", tick=True)
        if not result.get("auction_active"):
            break

    # 7. Final report
    print("\n=== FINAL STATE ===")
    final_ledger = result.get("ledger", {})
    for phone, balance in final_ledger.items():
        print(f"  {phone}: {balance} tokens remaining")
    print(f"  Tracks pushed to Spotify queue (mocked): {queued_tracks}")

    assert queued_tracks == ["spotify:track:mock_espresso_1"], "expected Espresso to be queued"
    assert final_ledger.get("+1-555-0200") == 50, "Bob (winner) should have 50 tokens left (100 - 50 bid)"
    assert final_ledger.get("+1-555-0100") == 100, "Alice (outbid) should be untouched at 100 tokens"
    print("\nWalkthrough completed and assertions passed.")


if __name__ == "__main__":
    run()
