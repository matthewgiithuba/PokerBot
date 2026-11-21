"""Heads-up Hold'em bot.

Plays a strategy trained offline by train.py (MCCFR on an abstracted game) and
falls back to a simple equity-vs-pot-odds rule whenever the situation is not in
the table. train.py imports this file, so the abstraction (hand buckets, action
names, info-set key) lives here and is shared by both.

Abstract actions:  f = fold   c = check/call   h = raise ~half pot
                   p = raise ~pot   a = all-in
"""

import json
import os
import random
from bisect import bisect
from itertools import combinations



STATS = {"last_done": -1, "faced": 0, "folds": 0, "acts": 0, "raises": 0}
SKIP = ("small_blind", "big_blind", "ante", "show")


def opstats(gs):
    me, cur, pending = gs["your_bot_id"], gs["hand_num"], None
    for e in gs["match_action_log"]:
        h = e["hand_num"]
        if h <= STATS["last_done"] or h >= cur:   # only finished, not-yet-counted hands
            continue
        a = e["action"]
        if a in SKIP:
            continue
        key = (h, e["street"])
        if e["bot_id"] == me:
            pending = key if a in ("raise", "all_in") else None
        else:
            if pending == key:
                STATS["faced"] += 1
                if a == "fold":
                    STATS["folds"] += 1
            pending = None
            STATS["acts"] += 1
            if a in ("raise", "all_in"):
                STATS["raises"] += 1
    STATS["last_done"] = max(STATS["last_done"], cur - 1)


def oprates():
    fold_rate = (STATS["folds"] + 5 * 0.40) / (STATS["faced"] + 5)
    aggr = (STATS["raises"] + 5 * 0.25) / (STATS["acts"] + 5)
    return fold_rate, aggr, STATS["faced"], STATS["acts"]






try:
    import eval7
except ImportError:  # only used when testing outside the sandbox
    eval7 = None

RANKS = "23456789TJQKA"
DECK = [r + s for r in RANKS for s in "shdc"]
STREETS = ["preflop", "flop", "turn", "river"]
BOARD_LEN = [0, 3, 4, 5]

ACTIONS = ["f", "c", "h", "p", "a"]
FRACS = {"h": 0.5, "p": 1.0}      # raise sizes as a fraction of the pot
MAX_RAISES = 3                     # raises allowed per street in the abstraction
EQ_CUTS = [0.35, 0.45, 0.52, 0.58, 0.65, 0.72, 0.82]   # -> 8 hand buckets
POST_SIMS = 400                   # Monte Carlo samples per postflop decision


# ── hand evaluation ─────────────────────────────────────────────────────────
_E7 = {}


def _card(c):
    x = _E7.get(c)
    if x is None:
        x = _E7[c] = eval7.Card(c)
    return x


def _rank5(cs):
    vs = sorted((RANKS.index(c[0]) + 2 for c in cs), reverse=True)
    flush = len({c[1] for c in cs}) == 1
    uniq = sorted(set(vs), reverse=True)
    straight = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight = 5
    cnt = {}
    for v in vs:
        cnt[v] = cnt.get(v, 0) + 1
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [g[1] for g in groups]
    order = [g[0] for g in groups]
    if straight and flush:
        return (8, straight)
    if shape == [4, 1]:
        return (7, *order)
    if shape == [3, 2]:
        return (6, *order)
    if flush:
        return (5, *vs)
    if straight:
        return (4, straight)
    if shape == [3, 1, 1]:
        return (3, *order)
    if shape == [2, 2, 1]:
        return (2, *order)
    if shape == [2, 1, 1, 1]:
        return (1, *order)
    return (0, *vs)


def score(cards):
    """Comparable strength of a 7-card hand (higher wins)."""
    if eval7 is not None:
        return eval7.evaluate([_card(c) for c in cards])
    return max(_rank5(c) for c in combinations(cards, 5))


def equity(hero, board, sims):
    """Win probability against a random hand (ties count half)."""
    known = set(hero) | set(board)
    deck = [c for c in DECK if c not in known]
    need = 5 - len(board)
    wins = 0.0
    for _ in range(sims):
        d = random.sample(deck, 2 + need)
        full = list(board) + d[2:]
        a = score(list(hero) + full)
        b = score(d[:2] + full)
        wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / sims


# ── preflop: 169 hand classes, equity cached (and saved by train.py) ────────
PRE_EQ = {}


def hand_class(hole):
    a, b = hole
    ra, rb = RANKS.index(a[0]), RANKS.index(b[0])
    if ra < rb:
        a, b = b, a
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def class_hand(cls):
    if len(cls) == 2:
        return [cls[0] + "s", cls[1] + "h"]
    return [cls[0] + "s", cls[1] + ("s" if cls[2] == "s" else "h")]


def all_classes():
    out = []
    for i, r1 in enumerate(reversed(RANKS)):
        for r2 in list(reversed(RANKS))[i:]:
            if r1 == r2:
                out.append(r1 + r2)
            else:
                out += [r1 + r2 + "s", r1 + r2 + "o"]
    return out


def preflop_eq(hole, sims=300):
    cls = hand_class(hole)
    v = PRE_EQ.get(cls)
    if v is None:
        v = PRE_EQ[cls] = equity(class_hand(cls), [], sims)
    return v


# ── abstraction shared with train.py ────────────────────────────────────────
def bucket_of(eq):
    return bisect(EQ_CUTS, eq)                      # 0..7


def owed_bucket(owed, pot):
    if owed <= 1e-9:
        return 0
    r = owed / pot if pot > 0 else 9.0
    return 1 + bisect([0.35, 0.7, 1.2], r)          # 1..4


def stack_bucket(bbs):
    return bisect([6, 12, 25, 50], bbs)             # 0..4


def info_key(street, pos, bucket, stk, owedb, hist):
    # hist is the check/call ('c') and raise ('r') sequence on this street only
    return "%d|%d|%d|%d|%d|%s" % (street, pos, bucket, stk, owedb, hist[-6:])


def raise_options(cur, my_bet, pot, min_to, max_to, nraises):
    """Legal abstract raises as {action: raise_to_total}. 'a' (all-in) is last."""
    out = {}
    if nraises >= MAX_RAISES or max_to <= cur + 1e-9:
        return out
    owed = cur - my_bet
    for a, fr in FRACS.items():
        t = max(min_to, cur + fr * (pot + owed))
        if t < max_to - 1e-9 and all(abs(t - v) > 1e-9 for v in out.values()):
            out[a] = t
    out["a"] = max_to
    return out


# ── load the trained table (read at import time only) ───────────────────────
STRATEGY = {}


def _load():
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "strategy.json")
        with open(path) as f:
            d = json.load(f)
        STRATEGY.update(d.get("strategy", {}))
        PRE_EQ.update(d.get("preflop_eq", {}))
    except Exception:
        pass


_load()


# ── playing ─────────────────────────────────────────────────────────────────
def _context(gs):
    street = STREETS.index(gs["street"])
    pos = 0 if gs["seat_to_act"] == gs["sb_seat"] else 1   # 0 = button/small blind
    bb = gs["blinds"]["bb"]
    my_stack, my_bet = gs["your_stack"], gs["your_bet_this_street"]
    opp = [p for p in gs["players"] if p.get("bot_id") != gs["your_bot_id"]][0]
    opp_stack, opp_bet = opp["stack"], opp.get("bet_this_street", 0)
    hist = ""
    cur_bet = 0
    for e in gs.get("action_log", []):
        if e.get("street") != gs["street"]:
            continue
        a, amt = e.get("action"), e.get("amount", 0) or 0
        if a in ("small_blind", "big_blind"):
            cur_bet = max(cur_bet, amt)
        elif a in ("check", "call"):
            hist += "c"
        elif a in ("raise", "bet", "all_in"):
            if amt > cur_bet:
                hist += "r"
                cur_bet = amt
            else:
                hist += "c"          # all-in that is only a call
    cur = gs["current_bet"]
    owed = gs["amount_owed"]
    pot = gs["pot"]
    hole, board = gs["your_cards"], gs["community_cards"]
    if street == 0:
        eq = preflop_eq(hole)
    else:
        eq = equity(hole, board, POST_SIMS)
    max_to = min(my_bet + my_stack, opp_bet + opp_stack)
    return dict(street=street, pos=pos, eq=eq, owed=owed, pot=pot, cur=cur,
                my_bet=my_bet, hist=hist, max_to=max_to,
                stk=stack_bucket(min(my_stack, opp_stack) / bb),
                min_to=gs["min_raise_to"], can_check=gs["can_check"])


def _equity_move(gs, ctx=None, note=""):
    eq = ctx["eq"] if ctx else equity(gs["your_cards"], gs["community_cards"], POST_SIMS)
    pot, owed = gs["pot"], gs["amount_owed"]
    top = gs["your_stack"] + gs["your_bet_this_street"]
    dbg = {"fallback": note, "eq": round(eq, 3)}
    if eq > 0.70:
        to = max(gs["min_raise_to"], gs["current_bet"] + int(pot * 0.7))
        if to >= top:
            return {"action": "all_in", "debug": dbg}
        return {"action": "raise", "amount": int(to), "debug": dbg}
    if gs["can_check"]:
        return {"action": "check", "debug": dbg}
    bet_size = owed / max(pot - owed, 1)
    if eq > owed / (pot + owed) + 0.03 + 0.07 * min(bet_size, 1.0):
        return {"action": "call", "debug": dbg}
    return {"action": "fold", "debug": dbg}


def _strategy_move(gs):
    c = _context(gs)
    key = info_key(c["street"], c["pos"], bucket_of(c["eq"]), c["stk"],
                   owed_bucket(c["owed"], c["pot"]), c["hist"])
    probs = STRATEGY.get(key)
    if probs is None:
        return _equity_move(gs, c, "no key " + key)

    try:
        opstats(gs)
        fold_rate, aggr, faced, acts = oprates()
    except Exception:
        fold_rate, aggr, faced, acts = 0.40, 0.25, 0, 0

    # shift the trained probabilities toward what this opponent actually does
    probs = list(probs)
    fi, ci, hi = ACTIONS.index("f"), ACTIONS.index("c"), ACTIONS.index("h")
    if aggr > 0.40:                              # bets loosely: fold less
        w = min((aggr - 0.40) / 0.25, 1.0) * min(acts, 40) / 40
        moved = probs[fi] * 0.6 * w
        probs[fi] -= moved
        probs[ci] += moved
    if fold_rate > 0.50 and c["owed"] <= 0:      # folds a lot: bet more when checked to
        w = min((fold_rate - 0.50) / 0.30, 1.0) * min(faced, 30) / 30
        moved = probs[ci] * 0.5 * w
        probs[ci] -= moved
        probs[hi] += moved

    raises = raise_options(c["cur"], c["my_bet"], c["pot"], c["min_to"],
                           c["max_to"], c["hist"].count("r"))
    legal = ["f", "c"] if c["owed"] > 0 else ["c"]
    legal += [a for a in ("h", "p", "a") if a in raises]
    weights = [probs[ACTIONS.index(a)] for a in legal]
    if sum(weights) <= 0:
        return _equity_move(gs, c, "no legal mass " + key)
    a = random.choices(legal, weights=weights)[0]

    # don't fold decent hands to small bets from an aggressive opponent
    if (a == "f" and c["street"] > 0 and acts >= 20 and aggr > 0.40
            and c["owed"] / max(c["pot"] - c["owed"], 1) < 0.5 and c["eq"] > 0.40):
        a = "c"

    dbg = {"key": key, "act": a, "eq": round(c["eq"], 3),
           "opp": [round(fold_rate, 2), round(aggr, 2), faced, acts]}

    if a == "f":
        return {"action": "check" if c["can_check"] else "fold", "debug": dbg}
    if a == "c":
        return {"action": "check" if c["can_check"] else "call", "debug": dbg}
    if a == "a":
        return {"action": "all_in", "debug": dbg}
    return {"action": "raise", "amount": int(round(raises[a])), "debug": dbg}

def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}
    try:
        return _strategy_move(game_state)
    except Exception as e:  # never crash: a crash folds
        try:
            return _equity_move(game_state, None, "error " + repr(e))
        except Exception:
            return {"action": "check" if game_state.get("can_check") else "fold"}