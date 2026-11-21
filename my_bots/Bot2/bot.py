import random
import eval7
import time

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
    
def monte(game_state):
    duration = 1.0 # Run for 0.6 seconds
    end_time = time.perf_counter() + duration

    if game_state["street"] == "preflop":
        return 0

    mycards = [eval7.Card(c) for c in game_state["your_cards"]]   # e.g. ["As", "Kh", "7d"]
    known = game_state["your_cards"] + game_state["community_cards"]
    deck = eval7.Deck()

    deck.cards = [c for c in deck.cards if str(c) not in known]

    board = [eval7.Card(c) for c in game_state["community_cards"]]
    
    mine = eval7.evaluate(mycards + board)

    win = lose = tie = total = 0
    while time.perf_counter() < end_time:
        deck1 = deck.cards[:] 
        random.shuffle(deck1)
        opponent = random.sample(deck1, k=2)
        theirs = eval7.evaluate(opponent + board)
        if mine > theirs:
            win += 1
        elif mine < theirs:
            lose += 1
        else:
            tie += 1
        total += 1
    return win/total if total else 0
def raise_to(gs, mult):
    amount = int(gs["min_raise_to"] * mult)
    max_total = gs["your_stack"] + gs["your_bet_this_street"]
    if amount >= max_total:
        return {"action": "all_in"}
    return {"action": "raise", "amount": amount}

def decide(game_state: dict) -> dict:
    if game_state.get("type") == "warmup":     # one call before hand 1; load things here if you like
        return {"action": "check"}

    # ── Your strategy goes here. This example is deliberately simple. ──────────
    cards = game_state["your_cards"]
    pot = game_state["pot"]
    owed = game_state["amount_owed"]
    street = game_state["street"]
    minraise = game_state["min_raise_to"]
    mychips = game_state["your_stack"]
    can_check = game_state["can_check"]

    if game_state["street"] == "preflop":
        equity = preflop_equity(game_state["your_cards"])
        if equity > 0.81:
            return raise_to(game_state, 2)
        elif equity > 0.61:
            if can_check:
                return {"action": "check"}
            elif mychips > owed:
                return {"action": "call"} 
            elif owed >= mychips:
                return {"action": "all_in"}  
        elif can_check:
            return {"action": "check"}

    elif game_state["street"] != "preflop":
        odds = monte(game_state)
        if odds > 0.97:
            return {"action": "all_in"}
        elif odds > 0.8:
            if minraise*2 < game_state["your_stack"]:
                araise = minraise*2
                return {"action": "raise", "amount": araise}
            elif can_check:
                if minraise < game_state["your_stack"]:
                    araise = minraise
                return {"action": "raise", "amount": araise} 
            else:
                return {"action": "all_in"}
        elif odds > 0.6:
            if minraise*1.5 < game_state["your_stack"]:
                araise = minraise*1.5
                return {"action": "raise", "amount": araise}
            elif can_check:
                return {"action": "check"}
            else:
                return {"action": "call"}
        elif odds < 0.6 and odds > 0.55:
            if can_check:
                return {"action": "check"}
    return {"action": "fold"}

        



