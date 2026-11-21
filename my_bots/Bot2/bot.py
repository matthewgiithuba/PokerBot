import random
import eval7
import time

STATS = {
    "last_done": -1,     # highest hand_num already counted
    "faced": 0,          # times I bet/raised and they responded
    "folds": 0,          # ...and they folded
    "acts": 0,           # their total actions
    "raises": 0,         # their raises/all-ins
}

def opstats(gs):
    me = gs["your_bot_id"]
    cur = gs["hand_num"]
    old = STATS["last_done"]

    new_hands = (cur - 1) - old
    if new_hands > 0:
        f = 0.99 ** new_hands
        for k in ("faced", "folds", "acts", "raises"):
            STATS[k] *= f

    pending = None
    for e in gs["match_action_log"]:
        h = e["hand_num"]
        if h <= STATS["last_done"] or h >= cur:
            continue
        a = e["action"]
        if a in ("small_blind", "big_blind", "ante", "show"):
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
    return fold_rate, aggr, STATS["faced"]

def last_opp_action(gs):
    me = gs["your_bot_id"]
    for e in reversed(gs["action_log"]):
        if e["bot_id"] != me and e["action"] not in ("small_blind", "big_blind", "ante", "show"):
            return e
    return None

PREFLOP = {
    "AA": 0.852, "AKs": 0.671, "AKo": 0.654, "AQs": 0.662, "AQo": 0.642, "AJs": 0.652,
    "AJo": 0.633, "ATs": 0.645, "ATo": 0.628, "A9s": 0.629, "A9o": 0.607, "A8s": 0.620,
    "A8o": 0.599, "A7s": 0.609, "A7o": 0.592, "A6s": 0.600, "A6o": 0.578, "A5s": 0.597,
    "A5o": 0.575, "A4s": 0.590, "A4o": 0.569, "A3s": 0.582, "A3o": 0.555, "A2s": 0.575,
    "A2o": 0.549, "KK": 0.823, "KQs": 0.636, "KQo": 0.613, "KJs": 0.622, "KJo": 0.606,
    "KTs": 0.615, "KTo": 0.597, "K9s": 0.599, "K9o": 0.578, "K8s": 0.581, "K8o": 0.557,
    "K7s": 0.577, "K7o": 0.550, "K6s": 0.567, "K6o": 0.545, "K5s": 0.556, "K5o": 0.531,
    "K4s": 0.551, "K4o": 0.522, "K3s": 0.540, "K3o": 0.512, "K2s": 0.532, "K2o": 0.505,
    "QQ": 0.800, "QJs": 0.603, "QJo": 0.580, "QTs": 0.597, "QTo": 0.568, "Q9s": 0.577,
    "Q9o": 0.550, "Q8s": 0.560, "Q8o": 0.536, "Q7s": 0.538, "Q7o": 0.516, "Q6s": 0.534,
    "Q6o": 0.510, "Q5s": 0.528, "Q5o": 0.501, "Q4s": 0.521, "Q4o": 0.488, "Q3s": 0.510,
    "Q3o": 0.478, "Q2s": 0.500, "Q2o": 0.473, "JJ": 0.774, "JTs": 0.576, "JTo": 0.553,
    "J9s": 0.558, "J9o": 0.533, "J8s": 0.540, "J8o": 0.516, "J7s": 0.522, "J7o": 0.498,
    "J6s": 0.508, "J6o": 0.479, "J5s": 0.500, "J5o": 0.471, "J4s": 0.494, "J4o": 0.461,
    "J3s": 0.480, "J3o": 0.452, "J2s": 0.473, "J2o": 0.445, "TT": 0.749, "T9s": 0.541,
    "T9o": 0.515, "T8s": 0.523, "T8o": 0.500, "T7s": 0.501, "T7o": 0.482, "T6s": 0.489,
    "T6o": 0.468, "T5s": 0.473, "T5o": 0.441, "T4s": 0.468, "T4o": 0.436, "T3s": 0.459,
    "T3o": 0.428, "T2s": 0.444, "T2o": 0.413, "99": 0.722, "98s": 0.507, "98o": 0.483,
    "97s": 0.492, "97o": 0.463, "96s": 0.479, "96o": 0.449, "95s": 0.458, "95o": 0.427,
    "94s": 0.437, "94o": 0.406, "93s": 0.434, "93o": 0.401, "92s": 0.426, "92o": 0.392,
    "88": 0.691, "87s": 0.479, "87o": 0.449, "86s": 0.461, "86o": 0.430, "85s": 0.445,
    "85o": 0.415, "84s": 0.429, "84o": 0.395, "83s": 0.406, "83o": 0.373, "82s": 0.404,
    "82o": 0.371, "77": 0.662, "76s": 0.456, "76o": 0.422, "75s": 0.434, "75o": 0.405,
    "74s": 0.420, "74o": 0.385, "73s": 0.399, "73o": 0.368, "72s": 0.381, "72o": 0.347,
    "66": 0.630, "65s": 0.428, "65o": 0.398, "64s": 0.414, "64o": 0.382, "63s": 0.395,
    "63o": 0.364, "62s": 0.381, "62o": 0.345, "55": 0.603, "54s": 0.415, "54o": 0.380,
    "53s": 0.397, "53o": 0.361, "52s": 0.379, "52o": 0.344, "44": 0.570, "43s": 0.386,
    "43o": 0.353, "42s": 0.367, "42o": 0.330, "33": 0.537, "32s": 0.359, "32o": 0.321,
    "22": 0.501,
}

RANKS = "AKQJT98765432"
def preflop_equity(cards):
    (r1, s1), (r2, s2) = cards
    hi, lo = sorted((r1, r2), key=RANKS.index)
    if hi == lo:
        return PREFLOP[hi + lo]
    return PREFLOP[hi + lo + ("s" if s1 == s2 else "o")]
def opp_cutoff(gs, aggr):
    """Minimum preflop-table strength their hand must have, given how they bet."""
    owed = gs["amount_owed"]
    if owed == 0:
        return 0.0                                   # they haven't bet: any hand
    size = owed / max(gs["pot"] - owed, 1)           # bet as a fraction of the pot before it
    cutoff = 0.40 + 0.15 * min(size, 1.5)            # bigger bet = stronger hand
    cutoff += {"flop": 0.0, "turn": 0.02, "river": 0.04}.get(gs["street"], 0.0)
    cutoff -= 0.20 * (aggr - 0.25)                   # aggressive players bet weaker hands
    return max(0.0, min(cutoff, 0.62))

def monte(game_state, cutoff, aggr=0.25, budget=0.5, target=2000):
    if game_state["street"] == "preflop":
        return 0
    hero = [eval7.Card(c) for c in game_state["your_cards"]]
    board = [eval7.Card(c) for c in game_state["community_cards"]]
    known = set(game_state["your_cards"] + game_state["community_cards"])
    remaining = [c for c in eval7.Deck().cards if str(c) not in known]
    need = 5 - len(board)
    late = game_state["street"] in ("turn", "river")

    win = tie = total = 0
    end_time = time.perf_counter() + budget
    while total < target and time.perf_counter() < end_time:
        drawn = random.sample(remaining, 2 + need)
        opp = drawn[:2]
        if cutoff and preflop_equity([str(opp[0]), str(opp[1])]) < cutoff:
            continue
        full_board = board + drawn[2:]
        mine = eval7.evaluate(hero + full_board)
        theirs = eval7.evaluate(opp + full_board)
        # passive players betting late usually have a pair or better
        if cutoff and late and aggr < 0.30 and eval7.handtype(theirs) == "High Card":
            continue
        if mine > theirs:
            win += 1
        elif mine == theirs:
            tie += 1
        total += 1
    return (win + tie / 2) / total if total else 0.5

def raise_to(gs, mult):
    amount = int(gs["min_raise_to"] * mult)
    max_total = gs["your_stack"] + gs["your_bet_this_street"]
    if amount >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": amount}
def bet_to(gs, frac):
    target = gs["current_bet"] + int(frac * (gs["pot"] + gs["amount_owed"]))
    target = max(target, gs["min_raise_to"])
    if target >= gs["your_stack"] + gs["your_bet_this_street"]:
        return {"action": "all_in"}
    return {"action": "raise", "amount": target}

def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":
        return {"action": "check"}

    fold_rate, aggr, faced = 0.40, 0.25, 0
    try:
        opstats(game_state)
        fold_rate, aggr, faced = oprates()
    except Exception:
        pass

    can_check = game_state["can_check"]
    pot = game_state["pot"]
    owed = game_state["amount_owed"]
    mychips = game_state["your_stack"]
    bb = game_state["blinds"]["bb"]
    preflop = game_state["street"] == "preflop"
    on_button = game_state["seat_to_act"] == game_state["button_seat"]

    evidence = min(faced, 30) / 30
    exploit = max(0.0, min((fold_rate - 0.40) / 0.30, 1.0)) * evidence

    if preflop:
        equity = preflop_equity(game_state["your_cards"])
        if game_state["current_bet"] > bb and equity < 0.75:
            equity -= 0.04 + 0.05 * min(owed / max(pot - owed, 1), 2.0)
    else:
        # CHANGE (#4, #6): pass aggr, and size the time budget from the deadline
        budget = min(0.4, game_state.get("time_limit_ms", 2000) / 1000 * 0.2)
        equity = monte(game_state, opp_cutoff(game_state, aggr), aggr, budget)

    bet_size = owed / max(pot - owed, 1)

    if mychips < 10 * bb and not can_check:
        return {"action": "all_in"} if equity > 0.52 else {"action": "fold"}

    bluff_freq = (0.20 if on_button else 0.08) + 0.35 * exploit
    if faced >= 15 and fold_rate < 0.25:
        bluff_freq = 0.0

    value_thr = 0.62 - 0.05 * exploit
    open_thr = 0.44 - 0.06 * exploit

    # CHANGE (#3): added `and equity < 0.80` so AA/KK fall through to the bigger value bet
    if preflop and on_button and game_state["current_bet"] <= bb and open_thr < equity < 0.80:
        return bet_to(game_state, 0.6)

    if not preflop and equity > 0.97:
        return {"action": "all_in"}
    if equity > 0.80:
        return bet_to(game_state, 0.75)
    if equity > value_thr and bet_size < 0.5:
        return bet_to(game_state, 0.5)

    if can_check:
        # CHANGE (#2): bluff only with true air (0.30), not hands that can win at showdown
        if not preflop and equity < 0.30 and random.random() < bluff_freq:
            return bet_to(game_state, 0.5)
        return {"action": "check"}

    call_amt = min(owed, mychips)
    pot_odds = call_amt / (pot + call_amt)
    margin = 0.05 + 0.10 * min(bet_size, 1.0)
    if preflop and owed < 3 * bb:
        margin -= 0.03
    if equity > pot_odds + margin:
        return {"action": "all_in"} if owed >= mychips else {"action": "call"}
    return {"action": "fold"}

        



