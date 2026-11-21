"""External-sampling MCCFR for heads-up Hold'em on the abstraction in bot.py.

Run from the repo root (venv active):
    python3 train.py --bot my_bots/matthew --minutes 30

Writes <bot>/data/strategy.json, which bot.py loads at import time.
Units are big blinds. Each iteration deals a fresh hand with a random stack depth.
"""

import argparse
import copy
import importlib.util
import json
import os
import random
import time

B = None            # the bot module, loaded in main()
NODES = {}          # info-set key -> [regret_sum[5], strategy_sum[5]]
IDX = {}
SIMS = 30           # equity samples per postflop bucket during training
STACKS_BB = [6, 10, 15, 25, 40, 60, 100]


class S:
    pass


def load_bot(path):
    spec = importlib.util.spec_from_file_location("pbot", os.path.join(path, "bot.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def new_hand():
    deck = B.DECK[:]
    random.shuffle(deck)
    s = S()
    s.hole = [deck[0:2], deck[2:4]]
    s.board = deck[4:9]
    eff = random.choice(STACKS_BB)
    s.stacks = [eff - 0.5, eff - 1.0]
    s.bets = [0.5, 1.0]
    s.invested = [0.5, 1.0]
    s.street = 0
    s.to_act = 0                 # small blind / button acts first preflop
    s.hist = ""
    s.nacts = 0
    s.nraises = 0
    s.last_raise = 1.0
    s.done = False
    s.payoff0 = 0.0
    s.cache = {}
    return s


def clone(s):
    n = copy.copy(s)
    n.stacks = s.stacks[:]
    n.bets = s.bets[:]
    n.invested = s.invested[:]
    return n


def legal(s):
    i, o = s.to_act, 1 - s.to_act
    cur = max(s.bets)
    owed = cur - s.bets[i]
    pot = sum(s.invested)
    acts = ["f", "c"] if owed > 1e-9 else ["c"]
    max_to = min(s.bets[i] + s.stacks[i], s.bets[o] + s.stacks[o])
    raises = B.raise_options(cur, s.bets[i], pot, cur + s.last_raise, max_to, s.nraises)
    acts += [a for a in ("h", "p", "a") if a in raises]
    return acts, raises, owed, pot


def step(s, a, raises):
    n = clone(s)
    i, o = n.to_act, 1 - n.to_act
    if a == "f":
        n.done = True
        n.payoff0 = -n.invested[0] if i == 0 else n.invested[1]
        return n
    if a == "c":
        owed = max(n.bets) - n.bets[i]
        n.stacks[i] -= owed
        n.bets[i] += owed
        n.invested[i] += owed
        n.hist += "c"
    else:
        t = raises[a]
        cur = max(n.bets)
        add = t - n.bets[i]
        n.stacks[i] -= add
        n.bets[i] = t
        n.invested[i] += add
        n.last_raise = t - cur
        n.nraises += 1
        n.hist += "r"
    n.nacts += 1
    if abs(n.bets[0] - n.bets[1]) < 1e-9 and n.nacts >= 2:
        if min(n.stacks) <= 1e-9 or n.street == 3:
            x = B.score(n.hole[0] + n.board)
            y = B.score(n.hole[1] + n.board)
            n.payoff0 = n.invested[1] if x > y else (-n.invested[0] if x < y else 0.0)
            n.done = True
        else:
            n.street += 1
            n.bets = [0.0, 0.0]
            n.nacts = 0
            n.nraises = 0
            n.last_raise = 1.0
            n.hist = ""
            n.to_act = 1         # big blind acts first after the flop
    else:
        n.to_act = o
    return n


def key_for(s, owed, pot):
    i = s.to_act
    cb = s.cache.get((i, s.street))
    if cb is None:
        if s.street == 0:
            eq = B.preflop_eq(s.hole[i])
        else:
            eq = B.equity(s.hole[i], s.board[:B.BOARD_LEN[s.street]], SIMS)
        cb = s.cache[(i, s.street)] = B.bucket_of(eq)
    return B.info_key(s.street, i, cb, B.stack_bucket(min(s.stacks)),
                      B.owed_bucket(owed, pot), s.hist)


def regret_strategy(node, acts):
    pos = [max(node[0][IDX[a]], 0.0) for a in acts]
    t = sum(pos)
    return [p / t for p in pos] if t > 0 else [1.0 / len(acts)] * len(acts)


def walk(s, trav):
    if s.done:
        return s.payoff0 if trav == 0 else -s.payoff0
    acts, raises, owed, pot = legal(s)
    key = key_for(s, owed, pot)
    node = NODES.get(key)
    if node is None:
        node = NODES[key] = [[0.0] * 5, [0.0] * 5]
    strat = regret_strategy(node, acts)
    if s.to_act == trav:
        vals = [walk(step(s, a, raises), trav) for a in acts]
        ev = sum(p * v for p, v in zip(strat, vals))
        for a, v in zip(acts, vals):
            node[0][IDX[a]] += v - ev
        return ev
    for a, p in zip(acts, strat):
        node[1][IDX[a]] += p
    a = random.choices(acts, weights=strat)[0]
    return walk(step(s, a, raises), trav)


def save(path, min_visits):
    strategy = {}
    for key, (_, ss) in NODES.items():
        t = sum(ss)
        if t >= min_visits:
            strategy[key] = [round(x / t, 4) for x in ss]
    os.makedirs(os.path.join(path, "data"), exist_ok=True)
    out = os.path.join(path, "data", "strategy.json")
    with open(out, "w") as f:
        json.dump({"strategy": strategy,
                   "preflop_eq": {k: round(v, 4) for k, v in B.PRE_EQ.items()}}, f)
    return out, len(strategy)


def main():
    global B, SIMS
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="my_bots/matthew")
    ap.add_argument("--iters", type=int, default=10**9)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--sims", type=int, default=30, help="equity samples per postflop bucket")
    ap.add_argument("--pre-sims", type=int, default=600, help="samples per preflop hand class")
    ap.add_argument("--min-visits", type=float, default=5)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    random.seed(a.seed)
    SIMS = a.sims
    B = load_bot(a.bot)
    IDX.update({x: i for i, x in enumerate(B.ACTIONS)})

    t0 = time.time()
    for cls in B.all_classes():
        if cls not in B.PRE_EQ:
            B.PRE_EQ[cls] = B.equity(B.class_hand(cls), [], a.pre_sims)
    print("preflop table done in %.1fs" % (time.time() - t0))

    t0 = time.time()
    total = 0.0
    it = 0
    while it < a.iters and time.time() - t0 < a.minutes * 60:
        total += walk(new_hand(), it % 2)
        it += 1
        if it % 500 == 0:
            print("iter %d  infosets %d  %.0fs" % (it, len(NODES), time.time() - t0), flush=True)
        if it % 5000 == 0:
            save(a.bot, a.min_visits)
    out, n = save(a.bot, a.min_visits)
    print("done: %d iterations, %d infosets saved -> %s" % (it, n, out))


if __name__ == "__main__":
    main()