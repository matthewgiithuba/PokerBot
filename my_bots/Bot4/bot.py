"""
Heads-up No-Limit Hold'em tournament bot (v1.1). Single file: numpy + stdlib only.

How it decides
--------------
* Hand evaluation: a vectorised numpy evaluator (thousands of hands per call).
* Preflop: an embedded 169x169 all-in equity table. Shoves, calls and 3-bets
  are expected-value calculations against the opponent's estimated range; below
  ~14bb it plays push/fold, with heads-up push/fold equilibrium ranges (also
  embedded) as the starting guess for how the opponent shoves and calls.
* Postflop: every opponent hand is evaluated over sampled run-outs, so we know
  each one's strength and our equity against any weighting of them. The
  opponent's range is narrowed from how often they take each action ("an action
  they take x% of the time comes from about their top x%, plus bluffs"), and we
  pick the action with the highest expected value using fold / call / raise
  rates learned from this opponent, by street and bet size.
* Opponent model: rebuilt every match from match_action_log, shrunk toward
  priors so it is sane from hand 1 and adapts within ~10 hands. Showdowns
  measure how closely the opponent's bets follow hand strength (kappa).
* Risk premium: against an opponent we out-play, deep-stacked all-ins need extra
  equity, because in a freezeout a coin flip for the match throws away our edge.
"""

import base64
import math
import random
import time
import zlib

import numpy as np

# =============================================================================
# Hand evaluator (cards are ints 0..51 = rank * 4 + suit)
# =============================================================================

RANKS = "23456789TJQKA"
SUITS = "cdhs"
_C5 = 13 ** 5
_AR13 = np.arange(13)


def _build_eval_tables():
    n = 1 << 13
    pop = np.zeros(n, np.int64)
    straight = np.full(n, -1, np.int64)
    top = {k: np.zeros(n, np.int64) for k in (1, 2, 3, 5)}
    for m in range(n):
        bits = [r for r in range(12, -1, -1) if (m >> r) & 1]
        pop[m] = len(bits)
        for k in (1, 2, 3, 5):
            v = 0
            for i in range(k):
                v = v * 13 + (bits[i] if i < len(bits) else 0)
            top[k][m] = v
        for t in range(12, 3, -1):
            w = 0b11111 << (t - 4)
            if m & w == w:
                straight[m] = t
                break
        else:
            if m & 0b1000000001111 == 0b1000000001111:
                straight[m] = 3
    return pop, straight, top[1], top[2], top[3], top[5]


_POP, _STRAIGHT, _TOP1, _TOP2, _TOP3, _TOP5 = _build_eval_tables()


def _hi(b):
    idx = 12 - np.argmax(b[:, ::-1], axis=1)
    return np.where(b.any(axis=1), idx, -1)


def _bit(x):
    return np.where(x >= 0, np.left_shift(1, np.maximum(x, 0)), 0)


def hand_values(cards):
    """cards: int array (N, k), 5 <= k <= 7. Returns int64 strengths (higher wins)."""
    cards = np.asarray(cards, dtype=np.int64)
    n = cards.shape[0]
    r = cards >> 2
    s = cards & 3
    cnt = np.bincount((r + 13 * np.arange(n)[:, None]).ravel(), minlength=13 * n).reshape(n, 13)
    rbit = np.left_shift(1, r)
    rmask = np.bitwise_or.reduce(rbit, axis=1)
    fmask = np.zeros(n, np.int64)
    for su in range(4):
        sm = np.bitwise_or.reduce(np.where(s == su, rbit, 0), axis=1)
        fmask = np.where(_POP[sm] >= 5, sm, fmask)
    has_fl = fmask > 0
    c2 = cnt >= 2
    q = _hi(cnt == 4)
    t = _hi(cnt >= 3)
    p1 = _hi(c2)
    p2 = _hi(c2 & (_AR13 != p1[:, None]))
    pfh = _hi(c2 & (_AR13 != t[:, None]))
    v = _TOP5[rmask]
    v = np.where(p1 >= 0, 1 * _C5 + p1 * 2197 + _TOP3[rmask & ~_bit(p1)], v)
    v = np.where(p2 >= 0, 2 * _C5 + p1 * 169 + p2 * 13 + _TOP1[rmask & ~_bit(p1) & ~_bit(p2)], v)
    v = np.where(t >= 0, 3 * _C5 + t * 169 + _TOP2[rmask & ~_bit(t)], v)
    st = _STRAIGHT[rmask]
    v = np.where(st >= 0, 4 * _C5 + st, v)
    v = np.where(has_fl, 5 * _C5 + _TOP5[fmask], v)
    v = np.where((t >= 0) & (pfh >= 0), 6 * _C5 + t * 13 + pfh, v)
    v = np.where(q >= 0, 7 * _C5 + q * 13 + _TOP1[rmask & ~_bit(q)], v)
    sf = _STRAIGHT[fmask]
    v = np.where(has_fl & (sf >= 0), 8 * _C5 + sf, v)
    return v


def card_int(s):
    return RANKS.index(s[0].upper()) * 4 + SUITS.index(s[1].lower())


# =============================================================================
# Embedded preflop tables (generated offline with the evaluator above)
#   _EQM_B64   169x169 all-in equity, class i vs class j, uint16 / 65535
#   _NASH_B64  heads-up push/fold equilibrium: for each class, the largest
#              stack (bb, x2) at which the SB shoves / the BB calls,
#              for ante 0 and ante 0.1bb (255 = always)
# Classes: pair rr -> r*13+r, suited hi/lo -> hi*13+lo, offsuit -> lo*13+hi
# =============================================================================

_EQM_B64 = (
    "eNocnH002/m+723jOI7lOo7rWq5jjDFtVdUY47i2iTRNIyIiz0+//PLLo4iINNVUU89CRERERBoRaYrGM1U1xti2bduO5VqO"
    "3emTMY5juZZlWZZlWa5tm7amc79zV1f7hzZZ+X3yfXi/35/Xpz469jB9UDWwM3xp+OrozFPTkHA4+On5SFrfajfna3cjyWhu"
    "7NPPmMIagg1pFnLDnPHQYOmaqvH72l6fZFitNxm0TScNMY1RDXuNkcajR6nVUeXvv5zSn9YG6tMMGw1v66+ZNxpqDXvdmKq5"
    "mpDS/0iO0Xkr8Q163bmx2JBmSDGa+ug6TjVciXpw8qVvRXBtWu0HA8Egrwuqu9Lzsi6tNl23V0nWIknL1a9rk3Umw5yRU1fv"
    "Ha8L1b+sC6jcfvBL6eCXGt1hLafuhd7c+LZbVkPU2XVblbgHL8sdDyKTPA98Kt9Vi2t3PRO1idXG6unqhJrAmr6K0vtzSSU1"
    "xgacPrrnqN5ZN1EbWOOsXqvUls9WvtQuJa3WaxuwvR79Nf2WLka3WRUBPveVMmKlo2bvq3XjT4/TaqOqX1SM1aTU4GpgHVQ1"
    "UK2qEdcOJw6zfXRrpUn5NomFi6MOUi6I1pyarKisjRtrpVsRlfdqpOPwOGOKGUqyZ2Kzp28t37pI9yneUf/8mV42JgoSIvxV"
    "KI7dw4uBEjnpLJbITxosSY0uFRj459x0VjxUAQVyp7gsJo6/Bb0RoIXjsUJOP3eWVcHCcZf4p2wUd5mxRUuFCWw5B38llonj"
    "hnGCODhuBDTI6M0NJ4+yYrkwK46cfjWMc8GOgYyMYXYEVEyezA7iRLB8WGPMy4z9LxTsFG4Pm8aO4w0TMXg7K4Dmw4qm1zDm"
    "GYuXklmjrGWOkBOVNYcTMjYYaexF2gJ7mkGjb106Yi2w06EFfNetBTaBfplRRj+jURizNDVl+4tzbjJLjN+4UcMQMsysC3oJ"
    "M5aZwoygrTPLLtvYJvTGN7O0V5QaRiBzn7nM8DADqeAT0Lujx7jXjD66eMW4tJYXwSojW0g2ojsbncPCBdX55JpLsSU+qhP5"
    "af6uxCsLkKXkXUgWkdFSP4018kKSCCvoQYzEnBniWTY6qxa/l7GUp1FBkvnoQ8FleBwW8yBeKXuCR+b4Mqv4KxKEq+Z0x5qh"
    "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"
    "1e9biVXK5wqaBG1NWkGye9m9BZPi9kf6bSIq/bYYpkXHFWAImHytadGENm7pegzvdV4dz67+lBeDJwLZwV2gbwiqD+ST+CZw"
    "g0zlxWDzv8CRZ0oJtDXGFo/Nfc+L5wZVL3KoOdHZ8yTbtWfUwkoCy9GwyJPVh9XyaqJrGPCttAMfMTMhkh5Uyall8rx5cXXh"
    "nLfsiDxlbg+JVXZC/d0/8MqWuS6eFw/L6a7jcjj4t7lelJcUBSR5KYkFz4L/Mv6XgNfWf57alG1E/Yvs2zfutKe3nr4Y/ezZ"
    "0k8zL++8PPOzfr3kb//8f6aerj81TqR65C7pkvpHxMql5+9fzL5B/eXv9wOe9nv+zhPxdH984sf+xY5/XHmGevHizZufP9vx"
    "e3LkQj/Vj8c7R169X9T/0+2F5ZXbr0Z+/n+2WE/Pf4t3Bzxtc4e+2noR9uxk6dY/Xfrpyeqbv/lObLkE3x65tlx/e4t+/WK5"
    "5blxCbuc9NN/+ivVddkd6g77dmlcvZG9EbahWL31+tLrq6BL6bW75ghry/jMmHbU40gbiRvpHVu2oK0yRdflt3dT7zGE2gdH"
    "o6/A+eSxyOzc0T4bUcntSIth9PY9IIrsIxHglWqyFdj97FIHVZd4h9L9u2sHd+fvqsfO2102jvnViALqdD9z053ZW/S7o1+E"
    "wHQ+jj3bvAXzqlG2dVCY6bcnuvfuvOv+v7HCdqlFbnthEgIFA+uS/MB19fBD6iIaztUj1VG3PtNv6lfM1NE37s++33zySc1U"
    "8ndrp7Rp8uU2njHM7G1zjRqn4t0hDYegPfryIzXJgx/lr9s/GpmmSVvbFNa123i6bpEnqF9RU9R25Zth7w6J8Xhk9/tL41ye"
    "s9ZZ682bNsxBP/WV0n9oovHYavUEjMzVdXFCue9q93TJQE+f0GNVIUpWHRq0P5+qLdYGe4O9MkYZSwaNsHikw37NRiB6dfDS"
    "Qm/MZuoyJ+yXjHzLJGi38Ra+JbuxKzwrPQFmnyUhoTfBnuUkcjbYCLZfeUDa3tfBkQf5cAsk6GG6X3dNF2cT9Ix+xrP0vet+"
    "+EQ46xcpInBjYjkTQBHqYSayEjM+XN8q9MO7LhuKz1eHcyBEx7zIKoSJa+GZyemd+F/gdnHli8C2RVOUbc2UYo93+o199ucl"
    "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"
)
_NASH_B64 = (
    "eNp1kltv4jAUhKUlvubq3ElCEpJCoVHbwC5qF/7/75o9jqDqQ3de7JFH8pxPB79WVg5TQXFbYTGM67DEKoRDYkKHFZzqAMaY"
    "kF7cwAl6IAu06yctHJa/AJssNmlHkYyecBybagRX/uY3OdzmE7wwLQ+4y+R1Pz5MWQ/z9WH2xxN+VssWcSHdqOevnN+N2XIB"
    "biWUF++EVBAkqf30yK2TUiod5JM1Gkq5XlT+pYByfXh+mNQQUtIsBiYpBgpo7QVxgXX7Dtf1/MBk1RP2FwRhFCdZ1U1/qE5W"
    "lFXTja9Lt/fz5VvTO1DhRsRtYehw6kncUiwjUM8WrJ0RUScdpD2caAesDdXJBzBez8C2LJJiR5OVO/vDc98eoNxou6C6fnwi"
    "NGX99vgzXXf7L7x1N31cH0in+fM/RAfGv7g9iTMX9k7cEoK4MLTlshdLTUrLKyze7EGIlPai9ckaD9olXA0sXS+EH5q8gyTM"
    "hBaE65kClIiyCs1wIeKEMSnbIyYgMnGaFZvxfLN7UNVtPx6WjcHle+l/sszqsg=="
)

_EQM = np.frombuffer(zlib.decompress(base64.b64decode(_EQM_B64)), dtype="<u2").astype(np.float64)
EQM = _EQM.reshape(169, 169) / 65535.0
_NASH = np.frombuffer(zlib.decompress(base64.b64decode(_NASH_B64)), dtype=np.uint8).astype(np.float64)
_NASH = _NASH.reshape(4, 169) / 2.0
_NASH[_NASH >= 127] = 999.0
PUSH_BB = {0: _NASH[0], 1: _NASH[2]}
CALL_BB = {0: _NASH[1], 1: _NASH[3]}

# =============================================================================
# Combos and classes
# =============================================================================

COMBOS = np.array([(a, b) for a in range(52) for b in range(a)], dtype=np.int64)
NC = len(COMBOS)
CARD_IN = np.zeros((52, NC), bool)
CARD_IN[COMBOS[:, 0], np.arange(NC)] = True
CARD_IN[COMBOS[:, 1], np.arange(NC)] = True


def cls_of(a, b):
    r1, r2 = a >> 2, b >> 2
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo or (a & 3) == (b & 3):
        return hi * 13 + lo
    return lo * 13 + hi


CLS = np.array([cls_of(a, b) for a, b in COMBOS], dtype=np.int64)
CLS_N = np.bincount(CLS, minlength=169).astype(np.float64)


def _class_info():
    hi = np.zeros(169, int)
    lo = np.zeros(169, int)
    suited = np.zeros(169, bool)
    names = [""] * 169
    for a in range(13):
        for b in range(13):
            i = a * 13 + b
            if a == b:
                hi[i] = lo[i] = a
                names[i] = RANKS[a] * 2
            elif a > b:
                hi[i], lo[i], suited[i] = a, b, True
                names[i] = RANKS[a] + RANKS[b] + "s"
            else:
                hi[i], lo[i] = b, a
                names[i] = RANKS[b] + RANKS[a] + "o"
    return hi, lo, suited, names


CL_HI, CL_LO, CL_SUITED, CL_NAME = _class_info()


def _compat():
    m = np.zeros((169, 169))
    for i in range(169):
        a, b = COMBOS[CLS == i][0]
        ok = ~(CARD_IN[a] | CARD_IN[b])
        m[i] = np.bincount(CLS[ok], minlength=169)
    return m


COMPAT = _compat()
EQR = (EQM * COMPAT).sum(1) / COMPAT.sum(1)          # equity vs a random hand


def _pct_from_score(score_cls):
    """Percentile of each class (1 = best), counting combos."""
    order = np.argsort(-score_cls, kind="stable")
    pct = np.zeros(169)
    above = 0.0
    for i in order:
        pct[i] = 1.0 - (above + 0.5 * CLS_N[i]) / NC
        above += CLS_N[i]
    return pct


_gap = CL_HI - CL_LO
PF_SCORE_CLS = (EQR + 0.03 * CL_SUITED + 0.02 * ((_gap == 1) & (CL_HI != CL_LO))
                + 0.01 * (_gap == 2) + 0.01 * ((CL_HI == 12) & CL_SUITED))
PF_PCT = _pct_from_score(PF_SCORE_CLS)               # by class, 1 = best
PF_SCORE = PF_SCORE_CLS[CLS]                          # by combo
PUSH_SCORE = {a: (np.minimum(PUSH_BB[a], 60) + 0.5 * EQR)[CLS] for a in (0, 1)}
CALL_SCORE = {a: (np.minimum(CALL_BB[a], 60) + 0.5 * EQR)[CLS] for a in (0, 1)}
SEVEN_TWO = {0 * 13 + 5, 5 * 13 + 0}                  # 72o, 72s


def nash_pct(table, eff_bb):
    return float(CLS_N[table >= eff_bb].sum() / NC)


# =============================================================================
# Small helpers
# =============================================================================

TAU = 0.07
# Risk premium (extra equity for stack-risking actions when deep): (per bb above 15bb, cap)
PREM_SHOVER = (0.0025, 0.22)
PREM_WEAK = (0.0015, 0.15)
PREM_UNKNOWN = (0.0012, 0.12)
DEEP_JAM_EQ = 0.62        # min equity (after premium) for an overbet shove above 40bb


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def lk_aggr(q, p, k):
    """Likelihood of a bet/raise taken with frequency p: mostly the top p."""
    p = min(max(p, 0.02), 0.98)
    return k * sig((q - (1 - p)) / TAU) + (1 - k) * p


def lk_call(q, f, r, k):
    """Likelihood of a call when they fold f and raise r: the middle band."""
    f = min(max(f, 0.0), 0.95)
    r = min(max(r, 0.0), 0.95 - f)
    return (k * sig((q - f) / TAU) * (1 - 0.7 * sig((q - (1 - r)) / TAU))
            + (1 - k) * max(1 - f - r, 0.03))


def lk_check(q, b, k):
    b = min(max(b, 0.02), 0.98)
    return k * (1 - 0.7 * sig((q - (1 - b)) / TAU)) + (1 - k) * (1 - b)


def wpct(score, w):
    """Percentile (0..1, 1 = strongest) of each item by score within weights w."""
    order = np.argsort(score, kind="stable")
    ws = w[order]
    cw = np.cumsum(ws)
    q = np.full(len(w), 0.5)
    tot = cw[-1] if len(cw) else 0.0
    if tot > 0:
        q[order] = (cw - 0.5 * ws) / tot
    return q


def shrink(k, n, prior, weight):
    return (k + prior * weight) / (n + weight)


def bucket(frac):
    if frac < 0.4:
        return 0
    if frac < 0.8:
        return 1
    if frac < 1.25:
        return 2
    if frac < 3.0:
        return 3
    return 4


BUCKET_FRAC = (0.3, 0.6, 1.0, 1.8, 5.0)


def size_curve(frac):
    """Prior shift of the fold rate with bet size (relative to a ~2/3-pot bet)."""
    if frac <= 1.25:
        return -0.07 + 0.14 * min(1.0, max(0.0, (frac - 0.3) / 0.95))
    return min(0.40, 0.07 + 0.08 * (frac - 1.25))


STREETS = ("flop", "turn", "river")
N_BOARD = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

# =============================================================================
# Replaying a hand from its action log
# =============================================================================


def walk(entries):
    """Replay one hand. Returns (events, shows, sb_id, final_state).
    Each event is a voluntary action with the context it was taken in."""
    pot = 0
    street = "preflop"
    cur = 0
    sbet = {}
    sb_id = None
    level = 0
    n_aggr = 0
    last_jam = False
    last_frac = 0.0
    events = []
    shows = {}
    for e in entries:
        a = e.get("action")
        who = e.get("bot_id")
        st = e.get("street")
        amt = e.get("amount") or 0
        if a == "show":
            shows[who] = e.get("cards")
            continue
        if st in STREETS and st != street:
            street, cur, sbet, n_aggr, last_jam, last_frac = st, 0, {}, 0, False, 0.0
        if a == "ante":
            pot += amt
            continue
        if a in ("small_blind", "big_blind"):
            pot += amt
            sbet[who] = sbet.get(who, 0) + amt
            cur = max(cur, sbet[who])
            if a == "small_blind":
                sb_id = who
            continue
        if a not in ("fold", "check", "call", "raise", "all_in"):
            continue
        mine = sbet.get(who, 0)
        to_call = max(0, cur - mine)
        ev = {"who": who, "street": street, "pot": pot, "to_call": to_call, "level": level,
              "n_aggr": n_aggr, "facing_jam": bool(last_jam and to_call > 0),
              "face_frac": last_frac if to_call > 0 else 0.0, "is_sb": who == sb_id, "frac": 0.0}
        if a == "fold":
            ev["type"] = "fold"
        elif a == "check":
            ev["type"] = "check"
        elif a == "call":
            ev["type"] = "call"
            pot += amt
            sbet[who] = mine + amt
        else:
            total = amt
            if total > cur:
                frac = (total - cur) / max(1.0, pot + to_call)
                if a == "all_in":
                    ev["type"] = "jam"
                elif street != "preflop" and cur == 0:
                    ev["type"] = "bet"
                else:
                    ev["type"] = "raise"
                ev["frac"] = frac
                pot += total - mine
                sbet[who] = total
                cur = total
                if street == "preflop":
                    level += 1
                n_aggr += 1
                last_jam = a == "all_in"
                last_frac = frac
            else:
                ev["type"] = "call"
                pot += max(0, total - mine)
                sbet[who] = max(mine, total)
        events.append(ev)
    return events, shows, sb_id, {"level": level, "n_aggr": n_aggr, "street": street}


# =============================================================================
# Opponent model (module state lasts exactly one match)
# =============================================================================

PF_PRIOR_D = {
    ("sb", 0): (0.22, 0.13, 0.62, 0.03),
    ("bb", 0): (0.00, 0.66, 0.31, 0.03),
    ("bb", 1): (0.33, 0.48, 0.16, 0.03),
    ("sb", 1): (0.35, 0.45, 0.14, 0.06),
    ("sb", 2): (0.45, 0.38, 0.10, 0.07),
    ("bb", 2): (0.45, 0.38, 0.10, 0.07),
}
STREET_BET_PRIOR = {"flop": 0.45, "turn": 0.40, "river": 0.35}
SIZE_RAISE_ADJ = (0.02, 0.0, -0.02, -0.04, -0.06)


class Model:
    def __init__(self):
        self.pf = {}            # (role, level, depth) -> [fold, call, raise, jam]
        self.vsjam = {"D": [0, 0], "S": [0, 0]}
        self.first = {}         # (street, opp_is_sb) -> [check, bet, big bet]
        self.face = {}          # (street, size bucket) -> [fold, call, raise]
        self.faceraise = {}     # street -> [fold, call, raise]
        self.done = set()
        self.depth = {}
        self.boards = {}
        self.last_depth = 200.0
        self.kpf = [0.0, 0.0]   # below-threshold count, expected count (preflop)
        self.kpo = [0.0, 0.0]   # same, postflop
        self.pf_dec_deep = 0
        self.pf_jam_deep = 0
        self.hands = 0
        self.showdowns = 0

    # ---------------------------------------------------------------- updating
    def update(self, gs, opp):
        cur_hand = gs.get("hand_num", 0)
        byhand = {}
        for e in gs.get("match_action_log") or []:
            h = e.get("hand_num")
            if h is None or h >= cur_hand or h in self.done:
                continue
            byhand.setdefault(h, []).append(e)
        for h in sorted(byhand):
            self.done.add(h)
            ents = byhand[h]
            acts = {e.get("action") for e in ents}
            if "small_blind" not in acts or "big_blind" not in acts:
                continue
            try:
                self._process(h, ents, opp)
            except Exception:
                pass

    def _process(self, h, ents, opp):
        depth = self.depth.get(h, self.last_depth)
        dcls = "S" if depth <= 20 else "D"
        events, shows, sb_id, _ = walk(ents)
        opp_sb = sb_id == opp
        self.hands += 1
        for ev in events:
            if ev["who"] != opp:
                continue
            t = ev["type"]
            if ev["street"] == "preflop":
                role = "sb" if ev["is_sb"] else "bb"
                c = self.pf.setdefault((role, min(ev["level"], 3), dcls), [0, 0, 0, 0])
                c[{"fold": 0, "check": 1, "call": 1, "raise": 2, "jam": 3}[t]] += 1
                if ev["facing_jam"] and t in ("fold", "call"):
                    self.vsjam[dcls][0 if t == "fold" else 1] += 1
                if dcls == "D":
                    self.pf_dec_deep += 1
                    self.pf_jam_deep += t == "jam"
            else:
                st = ev["street"]
                if ev["to_call"] == 0:
                    c = self.first.setdefault((st, opp_sb), [0, 0, 0])
                    if t == "check":
                        c[0] += 1
                    else:
                        c[1] += 1
                        c[2] += ev["frac"] >= 0.9
                elif ev["n_aggr"] <= 1:
                    c = self.face.setdefault((st, bucket(ev["face_frac"])), [0, 0, 0])
                    c[{"fold": 0, "call": 1}.get(t, 2)] += 1
                else:
                    c = self.faceraise.setdefault(st, [0, 0, 0])
                    c[{"fold": 0, "call": 1}.get(t, 2)] += 1
        cards = shows.get(opp)
        if cards:
            self.showdowns += 1
            self._kappa(h, events, cards, opp, opp_sb, dcls, depth)

    def _kappa(self, h, events, cards, opp, opp_sb, dcls, depth):
        hc = [card_int(c) for c in cards]
        q_pf = PF_PCT[cls_of(hc[0], hc[1])]
        boards = self.boards.get(h, {})
        for ev in events:
            if ev["who"] != opp or ev["type"] not in ("bet", "raise", "jam"):
                continue
            if ev["street"] == "preflop":
                role = "sb" if ev["is_sb"] else "bb"
                f = self.pf_freq(role, min(ev["level"], 3), dcls, depth, 0)
                p = f[2] + f[3]
                self.kpf[0] += q_pf < 1 - p
                self.kpf[1] += 1 - p
            else:
                board = boards.get(ev["street"])
                if not board:
                    continue
                pct = made_pct(hc, board)
                st = ev["street"]
                if ev["to_call"] == 0:
                    p = self.bet_freq(st, opp_sb, ev["frac"] >= 0.9)
                else:
                    p = self.raise_rate(st, bucket(ev["face_frac"]))
                self.kpo[0] += pct < 1 - p
                self.kpo[1] += 1 - p

    # ------------------------------------------------------------- estimates
    def pf_freq(self, role, lvl, dcls, eff_bb, ante):
        if dcls == "D":
            prior = PF_PRIOR_D.get((role, lvl), (0.50, 0.42, 0.0, 0.08))
            if eff_bb < 40:
                # open shoves become normal as stacks shrink toward 20bb
                j = max(prior[3], nash_pct(PUSH_BB[1 if ante else 0], eff_bb) * (40 - eff_bb) / 20)
                sc = (1 - j) / max(1e-9, 1 - prior[3])
                prior = (prior[0] * sc, prior[1] * sc, prior[2] * sc, j)
        else:
            push = nash_pct(PUSH_BB[1 if ante else 0], eff_bb)
            if (role, lvl) == ("sb", 0):
                prior = (max(0.05, 0.85 - push), 0.05, 0.10, push)
            elif (role, lvl) == ("bb", 0):
                prior = (0.0, 0.70, 0.10, 0.20)
            elif (role, lvl) == ("bb", 1):
                prior = (0.45, 0.30, 0.05, 0.20)
            else:
                prior = (0.50, 0.35, 0.0, 0.15)
        # pool this opponent's behaviour across similar spots, so a bot that
        # raises everything is recognised after a hand or two
        g = [0, 0, 0, 0]
        for (r_, l_, d_), v in self.pf.items():
            if (l_ >= 1) == (lvl >= 1) and d_ == dcls:
                for i in range(4):
                    g[i] += v[i]
        ng = sum(g)
        if ng:
            if lvl == 0 and role == "bb":
                g[0] = 0
                ng = sum(g)
            prior = tuple(shrink(g[i], ng, prior[i], 3.0) for i in range(4))
        c = self.pf.get((role, lvl, dcls), [0, 0, 0, 0])
        n = sum(c)
        out = [shrink(c[i], n, prior[i], 5.0) for i in range(4)]
        if dcls == "D" and self.pf_dec_deep:
            # an opponent that shoves a lot is recognised fast, whatever the spot
            jr = shrink(self.pf_jam_deep, self.pf_dec_deep, prior[3], 2.0)
            if jr > out[3]:
                sc = (1 - jr) / max(1e-9, 1 - out[3])
                out = [out[0] * sc, out[1] * sc, out[2] * sc, jr]
        return tuple(out)

    def jam_call(self, dcls, eff_bb, ante, need=None, open_jam=False):
        """How often they call an all-in. Prior: equilibrium for open shoves at
        short stacks, otherwise from the price they are getting."""
        if open_jam and dcls == "S":
            prior = nash_pct(CALL_BB[1 if ante else 0], eff_bb)
        elif need is not None:
            prior = min(0.92, max(0.12, 1.15 - 1.5 * need))
        else:
            prior = 0.35
        k = self.vsjam[dcls]
        return shrink(k[1], k[0] + k[1], prior, 4.0)

    def bet_freq(self, street, opp_sb, big=False):
        tb = sum(v[1] for v in self.first.values())
        tn = sum(v[0] + v[1] for v in self.first.values())
        g = shrink(tb, tn, 0.42, 6.0)
        k = self.first.get((street, opp_sb), [0, 0, 0])
        n = k[0] + k[1]
        p = shrink(k[1], n, min(0.95, max(0.05, g + STREET_BET_PRIOR[street] - 0.42)), 5.0)
        if big:
            return shrink(k[2], n, p * 0.35, 5.0)
        return p

    def _face_totals(self, street=None):
        f = n = 0
        for (st, _), v in self.face.items():
            if street is None or st == street:
                f += v[0]
                n += sum(v)
        return f, n

    def fold_rate(self, street, b, frac=None):
        f, n = self._face_totals()
        g = shrink(f, n, 0.40, 6.0)
        f, n = self._face_totals(street)
        s = shrink(f, n, g + (0.04 if street == "river" else 0.0), 6.0)
        frac = BUCKET_FRAC[b] if frac is None else frac
        prior = min(0.95, max(0.01, s + size_curve(frac)))
        k = self.face.get((street, b), [0, 0, 0])
        return shrink(k[0], sum(k), prior, 4.0)

    def fold_pair(self, street, b, frac):
        """(fold rate for fold equity, fold rate for the calling range): pessimistic on
        both sides while there is little data in this size bucket."""
        f = self.fold_rate(street, b, frac)
        k = self.face.get((street, b), [0, 0, 0])
        d = 0.10 * 4.0 / (sum(k) + 4.0)
        return max(0.0, f - d), min(0.97, f + d)

    def raise_rate(self, street, b):
        r = sum(v[2] for v in self.face.values())
        n = sum(sum(v) for v in self.face.values())
        g = shrink(r, n, 0.10, 6.0)
        k = self.face.get((street, b), [0, 0, 0])
        return shrink(k[2], sum(k), min(0.9, max(0.01, g + SIZE_RAISE_ADJ[b])), 4.0)

    def vs_raise(self, street, frac=0.9):
        tot = [0, 0, 0]
        for v in self.faceraise.values():
            for i in range(3):
                tot[i] += v[i]
        n = sum(tot)
        gf = shrink(tot[0], n, 0.35, 4.0)
        gr = shrink(tot[2], n, 0.08, 4.0)
        k = self.faceraise.get(street, [0, 0, 0])
        f = shrink(k[0], sum(k), gf, 3.0)
        f = min(0.95, max(0.02, f + size_curve(frac) - size_curve(0.9)))
        return f, shrink(k[2], sum(k), gr, 3.0), 0.10 * 4.0 / (n + 4.0)

    def kappa_pf(self):
        return float(np.clip(1 - (self.kpf[0] + 0.1 * 5) / (self.kpf[1] + 5), 0.15, 0.95))

    def kappa_post(self):
        return float(np.clip(1 - (self.kpo[0] + 0.25 * 5) / (self.kpo[1] + 5), 0.15, 0.95))

    def flags(self):
        f, n = self._face_totals()
        fold_post = shrink(f, n, 0.4, 3.0)
        tb = sum(v[1] for v in self.first.values())
        tn = sum(v[0] + v[1] for v in self.first.values())
        pfa = sum(v[2] + v[3] for v in self.pf.values())
        pfn = sum(sum(v) for v in self.pf.values())
        out = {
            "shover": self.pf_dec_deep >= 3 and self.pf_jam_deep / self.pf_dec_deep >= 0.45,
            "pf_aggro": pfn >= 6 and pfa / pfn >= 0.75,
            "station": n >= 6 and fold_post <= 0.15,
            "nit": n >= 6 and fold_post >= 0.6,
            "maniac": tn >= 8 and tb / tn >= 0.8,
            "random": self.kpo[1] >= 3 and self.kappa_post() < 0.35,
        }
        out["weak"] = (out["shover"] or out["station"] or out["maniac"] or out["random"]
                       or out["pf_aggro"])
        return out


MODEL = Model()


def made_pct(hand, board):
    """Percentile of a known hand's current strength among all other hands."""
    used = list(hand) + list(board)
    act = np.nonzero(~CARD_IN[used].any(0))[0]
    b = np.array(board, dtype=np.int64)
    vals = hand_values(np.concatenate([COMBOS[act], np.tile(b, (len(act), 1))], axis=1))
    mine = hand_values(np.array([list(hand) + list(board)]))[0]
    return float(((vals < mine).sum() + 0.5 * (vals == mine).sum()) / max(1, len(vals)))


# =============================================================================
# Per-street strength matrix (cached per hand)
# =============================================================================

_CACHE = {"hand": None, "streets": {}}
_SPEED = {"evals_per_s": 4.0e5}
_PAIRS47 = {}


def street_data(our, board, budget_s=0.12):
    key = (tuple(our), tuple(board))
    hit = _CACHE["streets"].get(key)
    if hit is not None:
        return hit
    used = list(our) + list(board)
    act = np.nonzero(~CARD_IN[used].any(0))[0]
    rest = np.array([c for c in range(52) if c not in set(used)], dtype=np.int64)
    need = 5 - len(board)
    A = len(act)
    nrun = int(min(48, max(8, _SPEED["evals_per_s"] * budget_s / max(1, A))))
    if need == 0:
        runs = np.zeros((1, 0), dtype=np.int64)
    elif need == 1:
        runs = rest[:, None]
        if len(runs) > nrun:
            runs = runs[np.sort(np.random.choice(len(runs), nrun, replace=False))]
    else:
        n = len(rest)
        if n not in _PAIRS47:
            _PAIRS47[n] = np.array([(i, j) for i in range(n) for j in range(i)], dtype=np.int64)
        pairs = _PAIRS47[n]
        pick = np.random.choice(len(pairs), min(nrun, len(pairs)), replace=False)
        runs = rest[pairs[pick]]
    R = runs.shape[0]
    t_start = time.perf_counter()
    b = np.array(board, dtype=np.int64).reshape(1, -1)
    full = np.concatenate([np.repeat(COMBOS[act], R, axis=0),
                           np.repeat(b, A * R, axis=0),
                           np.tile(runs, (A, 1))], axis=1)
    vals = hand_values(full).reshape(A, R)
    invalid = np.zeros((A, R), bool)
    for k in range(need):
        invalid |= CARD_IN[runs[:, k]][:, act].T
    valid = ~invalid
    ours = hand_values(np.concatenate([np.repeat(np.array(our, dtype=np.int64).reshape(1, 2), R, 0),
                                       np.repeat(b, R, axis=0), runs], axis=1))
    acc = np.zeros(A)
    for r in range(R):
        m = valid[:, r]
        sv = np.sort(vals[m, r])
        lo = np.searchsorted(sv, vals[:, r], "left")
        hi = np.searchsorted(sv, vals[:, r], "right")
        acc += np.where(m, (lo + 0.5 * (hi - lo)) / max(1, len(sv)), 0.0)
    vcount = valid.sum(1)
    strength = acc / np.maximum(vcount, 1)
    win = ((ours[None, :] > vals) & valid).sum(1) + 0.5 * ((ours[None, :] == vals) & valid).sum(1)
    data = {"act": act, "strength": strength, "s": win.astype(np.float64),
            "v": vcount.astype(np.float64), "R": R}
    if A * R >= 5000:
        # keep the speed estimate honest with what this machine is actually doing
        real = A * R / max(1e-4, time.perf_counter() - t_start)
        _SPEED["evals_per_s"] = min(3.0e6, max(1.0e5, 0.7 * _SPEED["evals_per_s"] + 0.3 * real))
    _CACHE["streets"][key] = data
    return data


def eq_of(sd, wa):
    den = float(wa @ sd["v"])
    if den <= 0:
        return 0.5
    return float(wa @ sd["s"]) / den


# =============================================================================
# Decision context
# =============================================================================


class Ctx:
    def __init__(self, gs):
        self.gs = gs
        self.me = gs["your_bot_id"]
        players = gs.get("players") or []
        opp = [p for p in players if p.get("bot_id") != self.me]
        self.opp_p = opp[0] if opp else {"bot_id": "?", "stack": 0, "bet_this_street": 0}
        self.opp = self.opp_p.get("bot_id")
        bl = gs.get("blinds") or {}
        self.bb = max(1, int(bl.get("bb") or 100))
        self.sb = int(bl.get("sb") or self.bb // 2)
        self.ante = int(bl.get("ante") or 0)
        self.pot = int(gs.get("pot") or 0)
        self.owed = int(gs.get("amount_owed") or 0)
        self.can_check = bool(gs.get("can_check", self.owed == 0))
        self.my_stack = int(gs.get("your_stack") or 0)
        self.my_bet = int(gs.get("your_bet_this_street") or 0)
        self.opp_stack = int(self.opp_p.get("stack") or 0)
        self.opp_bet = int(self.opp_p.get("bet_this_street") or 0)
        self.cur = int(gs.get("current_bet") or 0)
        self.min_to = int(gs.get("min_raise_to") or (self.cur + self.bb))
        self.street = gs.get("street", "preflop")
        self.we_sb = gs.get("seat_to_act") == gs.get("sb_seat")
        self.ip = self.we_sb
        self.our = [card_int(c) for c in gs["your_cards"]]
        self.board = [card_int(c) for c in gs.get("community_cards") or []]
        self.cls = cls_of(self.our[0], self.our[1])
        self.is72 = self.cls in SEVEN_TWO
        self.eff_total = min(self.my_stack + self.my_bet, self.opp_stack + self.opp_bet)
        self.eff_bb = self.eff_total / self.bb
        self.dcls = "S" if self.eff_bb <= 20 else "D"
        self.ante_key = 1 if self.ante > 0 else 0
        self.flags = MODEL.flags()

    def premium(self, risk):
        """Extra equity required for a big, stack-risking action (see module doc)."""
        if self.eff_bb <= 15:
            return 0.0
        frac = risk / max(1.0, self.my_stack + self.my_bet)
        if frac < 0.3:
            return 0.0
        if self.flags["shover"]:
            slope, cap = PREM_SHOVER
        elif self.flags["weak"]:
            slope, cap = PREM_WEAK
        else:
            slope, cap = PREM_UNKNOWN
        base = min(cap, slope * (self.eff_bb - 15))
        return base * min(1.0, frac / 0.8)

    # actions ---------------------------------------------------------------
    def fold(self):
        return {"action": "check"} if self.can_check else {"action": "fold"}

    def check_call(self):
        return {"action": "check"} if self.can_check else {"action": "call"}

    def raise_to(self, total):
        if self.opp_stack <= 0:
            return self.check_call()
        cap_me = self.my_stack + self.my_bet
        if cap_me <= self.cur:
            return self.check_call()
        total = int(min(int(total), self.opp_stack + self.opp_bet))
        total = max(total, self.min_to)
        if total >= cap_me:
            return {"action": "all_in"}
        return {"action": "raise", "amount": int(total)}

    def jam(self):
        if self.opp_stack <= 0:
            return self.check_call()
        return self.raise_to(self.my_stack + self.my_bet)


# =============================================================================
# Opponent range for the current hand
# =============================================================================


def opp_range(c, events, opp_sb):
    w = np.ones(NC)
    w[CARD_IN[c.our].any(0)] = 0.0
    kpf = MODEL.kappa_pf()
    kpo = MODEL.kappa_post()
    for ev in events:
        if ev["who"] != c.opp or ev["street"] != "preflop":
            continue
        t = ev["type"]
        role = "sb" if ev["is_sb"] else "bb"
        fo, ca, ra, ja = MODEL.pf_freq(role, min(ev["level"], 3), c.dcls, c.eff_bb, c.ante)
        if ev["facing_jam"] and t == "call":
            p = MODEL.jam_call(c.dcls, c.eff_bb, c.ante, 0.45, open_jam=ev["level"] <= 1)
            w = w * lk_aggr(wpct(CALL_SCORE[c.ante_key], w), p, kpf)
        elif t == "jam":
            score = PUSH_SCORE[c.ante_key] if c.eff_bb <= 30 else PF_SCORE
            w = w * lk_aggr(wpct(score, w), ja, kpf)
        elif t == "raise":
            w = w * lk_aggr(wpct(PF_SCORE, w), ra + ja, kpf)
        elif t in ("call", "check"):
            w = w * lk_call(wpct(PF_SCORE, w), fo, ra + ja, kpf)
    if w.sum() <= 0:
        w = np.ones(NC)
        w[CARD_IN[c.our].any(0)] = 0.0
    if c.street == "preflop":
        return w, None
    sd_now = None
    for st in STREETS:
        if N_BOARD[st] > len(c.board):
            break
        board = c.board[:N_BOARD[st]]
        w[CARD_IN[board].any(0)] = 0.0
        evs = [ev for ev in events if ev["street"] == st and ev["who"] == c.opp]
        sd = None
        if evs or st == c.street:
            sd = street_data(c.our, board, 0.12 if st == c.street else 0.05)
        if st == c.street:
            sd_now = sd
        if not evs:
            continue
        for ev in evs:
            wa = w[sd["act"]]
            if wa.sum() <= 0:
                break
            q = wpct(sd["strength"], wa)
            t = ev["type"]
            if ev["to_call"] == 0:
                if t == "check":
                    L = lk_check(q, MODEL.bet_freq(st, opp_sb), kpo)
                else:
                    L = lk_aggr(q, MODEL.bet_freq(st, opp_sb, ev["frac"] >= 0.9), kpo)
            elif ev["n_aggr"] <= 1:
                bk = bucket(ev["face_frac"])
                f, r = MODEL.fold_rate(st, bk, ev["face_frac"]), MODEL.raise_rate(st, bk)
                L = lk_call(q, f, r, kpo) if t == "call" else lk_aggr(q, r, kpo)
            else:
                f, r, _ = MODEL.vs_raise(st, ev["face_frac"])
                L = lk_call(q, f, r, kpo) if t == "call" else lk_aggr(q, r, kpo)
            w[sd["act"]] = wa * L
    return w, sd_now


# =============================================================================
# Preflop
# =============================================================================


def eq_pf(c, w):
    tot = w.sum()
    if tot <= 0:
        return float(EQR[c.cls])
    return float(np.bincount(CLS, w, minlength=169) @ EQM[c.cls] / tot)


def jam_ev(c, w, level=0):
    """EV (chips, relative to folding) of moving all-in now."""
    T = c.eff_total
    X = T - c.my_bet
    C = T - c.opp_bet
    need = C / max(1.0, c.pot + X + C)
    cj = MODEL.jam_call(c.dcls, c.eff_bb, c.ante, need, open_jam=level == 0)
    cont = w * lk_aggr(wpct(CALL_SCORE[c.ante_key], w), cj, MODEL.kappa_pf())
    eqc = eq_pf(c, cont)
    prem = c.premium(X)
    if c.eff_bb > 40 and X > 4 * c.pot and eqc - prem < DEEP_JAM_EQ:
        return -1e9          # deep overbet shoves only with a clear equity edge
    return (1 - cj) * c.pot + cj * ((eqc - prem) * (c.pot + X + C) - X)


def preflop(c, w, level, dbg):
    q_us = PF_PCT[c.cls]
    short = c.eff_bb <= (15 if c.ante else 14)
    dbg["q"] = round(float(q_us), 3)

    # --- button, first to act ------------------------------------------------
    if c.we_sb and level == 0 and c.owed > 0:
        if short:
            if c.is72 and c.eff_bb <= 12:
                dbg["why"] = "72 shove"
                return c.jam()
            ev = jam_ev(c, w)
            dbg["ev_jam"] = round(ev / c.bb, 2)
            return c.jam() if ev > 0.1 * c.bb else c.fold()
        fo, ca, ra, ja = MODEL.pf_freq("bb", 1, c.dcls, c.eff_bb, c.ante)
        n_bb1 = sum(MODEL.pf.get(("bb", 1, c.dcls), [0, 0, 0, 0]))
        if c.eff_bb > 40:
            open_pct, size = 0.82, 2.5
        elif c.eff_bb > 25:
            open_pct, size = 0.75, 2.2
        else:
            open_pct, size = 0.65, 2.0
        strong = q_us >= 0.80
        if n_bb1 >= 3 and ja >= 0.40:
            # they shove over opens: open only what can call a shove
            jr = np.ones(NC)
            jr[CARD_IN[c.our].any(0)] = 0
            jr = jr * lk_aggr(wpct(PF_SCORE, jr), ra + ja, MODEL.kappa_pf())
            eq = eq_pf(c, jr)
            need = 0.5 - 0.5 * c.pot / (c.eff_total * 2) + c.premium(c.eff_total)
            dbg["vs_shover"] = round(eq, 3)
            return c.raise_to(2 * c.bb) if eq >= need else c.fold()
        if n_bb1 >= 4 and fo >= 0.55:
            open_pct, size = 1.0, 2.0
        elif n_bb1 >= 6 and fo <= 0.15:
            open_pct = 0.60
            size = 3.0 if strong else 2.5
        if c.is72 and not c.flags["station"]:
            dbg["why"] = "72 open"
            return c.raise_to(2.2 * c.bb)
        if q_us >= 1 - open_pct:
            return c.raise_to(int(size * c.bb))
        return c.fold()

    # --- big blind after a limp ----------------------------------------------
    if c.owed == 0:
        eq = eq_pf(c, w)
        if short:
            ev_jam = jam_ev(c, w)
            ev_chk = 0.85 * eq * c.pot
            dbg["ev_jam"], dbg["ev_chk"] = round(ev_jam / c.bb, 2), round(ev_chk / c.bb, 2)
            return c.jam() if ev_jam > ev_chk else {"action": "check"}
        fo, ca, ra, ja = MODEL.pf_freq("sb", 1, c.dcls, c.eff_bb, c.ante)
        n_sb1 = sum(MODEL.pf.get(("sb", 1, c.dcls), [0, 0, 0, 0]))
        iso_pct, size = 0.35, 3.5
        if n_sb1 >= 3 and fo >= 0.5:
            iso_pct, size = 0.6, 3.0
        elif c.flags["station"] or (n_sb1 >= 3 and fo <= 0.15):
            iso_pct, size = 0.28, 4.5
        if q_us >= 1 - iso_pct or (c.is72 and not c.flags["station"]):
            return c.raise_to(int(size * c.bb))
        return {"action": "check"}

    # --- facing a raise --------------------------------------------------------
    return facing_raise_pf(c, w, level, dbg)


def facing_raise_pf(c, w, level, dbg):
    eq0 = eq_pf(c, w)
    call_cost = min(c.owed, c.my_stack)
    pot_eff = c.pot - max(0, c.owed - c.my_stack)
    commit = call_cost >= c.my_stack or c.opp_stack <= 0
    dbg["eq"] = round(eq0, 3)
    if commit:
        prem = c.premium(call_cost)
        ev_call = (eq0 - prem) * (pot_eff + call_cost) - call_cost
        dbg["ev_call"] = round(ev_call / c.bb, 2)
        return c.check_call() if ev_call > 0 else c.fold()

    pot_after = c.pot + call_cost
    left = c.my_stack - call_cost
    committed = left < 0.6 * pot_after
    if committed:
        rc, prem = 1.0, c.premium(c.my_stack)
    else:
        rc, prem = (0.93 if c.we_sb else 0.80), 0.0
    opts = [("fold", 0.0, None), ("call", rc * (eq0 - prem) * pot_after - call_cost, None)]

    # all-in
    opts.append(("jam", jam_ev(c, w, level), None))

    # a non-all-in raise when deep enough
    y = int((3.2 if level == 1 else 2.4) * c.cur)
    add = y - c.my_bet
    if level <= 2 and add < 0.4 * (c.my_stack + c.my_bet) and y > c.cur:
        opp_role = "bb" if c.we_sb else "sb"
        fo, ca, ra, ja = MODEL.pf_freq(opp_role, min(level + 1, 3), c.dcls, c.eff_bb, c.ante)
        kpf = MODEL.kappa_pf()
        q = wpct(PF_SCORE, w)
        callw = w * lk_call(q, fo, ra + ja, kpf)
        rrw = w * lk_aggr(q, ra + ja, kpf)
        their_add = y - c.cur
        r3 = 0.92 if c.we_sb else 0.86
        ev_called = r3 * eq_pf(c, callw) * (c.pot + add + their_add) - add
        T = c.eff_total
        dead = c.pot - c.my_bet - c.opp_bet
        ev_rr = max(-add, (eq_pf(c, rrw) - c.premium(T)) * (dead + 2 * T) - (T - c.my_bet))
        cc = max(0.0, 1 - fo - ra - ja)
        ev = fo * c.pot + cc * ev_called + (ra + ja) * ev_rr
        if c.is72 and not c.flags["station"]:
            ev += 0.3 * c.bb
        opts.append(("raise", ev, y))

    best = max(opts, key=lambda o: o[1])
    dbg["opts"] = {o[0]: round(o[1] / c.bb, 2) for o in opts}
    if best[0] == "fold":
        return c.fold()
    if best[0] == "call":
        return c.check_call()
    if best[0] == "jam":
        return c.jam()
    return c.raise_to(best[2])


# =============================================================================
# Postflop
# =============================================================================

SIZES = {"flop": (0.33, 0.66, 1.0), "turn": (0.5, 0.75, 1.2), "river": (0.5, 0.8, 1.2)}
REAL = {"flop": 0.88, "turn": 0.94, "river": 1.0}


def postflop(c, w, sd, opp_sb, dbg):
    act = sd["act"]
    wa = w[act]
    if wa.sum() <= 0:
        wa = np.ones(len(act))
    eq0 = eq_of(sd, wa)
    eq_rand = eq_of(sd, np.ones(len(act)))
    q = wpct(sd["strength"], wa)
    kap = MODEL.kappa_post()
    P = max(1, c.pot)
    st = c.street
    R = 1.0 if (c.ip or st == "river") else REAL[st]
    my, opp_rem = c.my_stack, c.opp_stack
    bluffish = eq_rand < 0.40
    bonus = (0.01 * P if bluffish else 0.0) + (0.04 * P if (c.is72 and bluffish) else 0.0)
    dbg.update(eq=round(eq0, 3), eq_rand=round(eq_rand, 3), R=sd["R"])
    opts = []
    if c.owed == 0:
        opts.append(("check", R * eq0 * P, None))
        if opp_rem > 0 and my > 0:
            maxbet = min(my, opp_rem)
            cands = set(int(round(s * P)) for s in SIZES[st])
            spr = maxbet / P
            n_left = {"flop": 3, "turn": 2, "river": 1}[st]
            geo = ((1 + 2 * spr) ** (1.0 / n_left) - 1) / 2
            if 0.25 <= geo <= 1.5:
                cands.add(int(round(geo * P)))
            if spr <= 2.5 or (st == "river" and spr <= 3.5):
                cands.add(maxbet)
            seen = set()
            for B in sorted(cands):
                B = int(min(max(B, c.bb, c.min_to), maxbet))
                if B in seen or B <= 0:
                    continue
                seen.add(B)
                bk = bucket(B / P)
                allin = B >= maxbet
                f, f_hi = MODEL.fold_pair(st, bk, B / P)
                r = 0.0 if B >= opp_rem else MODEL.raise_rate(st, bk)
                eqc = eq_of(sd, wa * lk_call(q, f_hi, r, kap))
                rb = 1.0 if (allin or st == "river") else R
                ev_called = rb * (eqc - c.premium(B)) * (P + 2 * B) - B
                ev_raised = 0.0
                if r > 0:
                    X = int(min(max(3 * B, B + c.bb), opp_rem, my))
                    eqr_ = eq_of(sd, wa * lk_aggr(q, r, kap))
                    ev_raised = max(-B, (eqr_ - c.premium(X)) * (P + 2 * X) - X)
                ev = f * P + max(0.0, 1 - f - r) * ev_called + r * ev_raised + bonus
                opts.append(("bet", ev, B))
    else:
        call_cost = min(c.owed, my)
        pot_eff = P - max(0, c.owed - my)
        allin_call = call_cost >= my or opp_rem <= 0
        rc = 1.0 if (allin_call or st == "river") else R
        opts.append(("fold", 0.0, None))
        opts.append(("call", rc * (eq0 - c.premium(call_cost)) * (pot_eff + call_cost) - call_cost, None))
        if opp_rem > 0 and my > call_cost:
            kinds = ("pot", "jam") if c.n_aggr <= 2 else ("jam",)
            for kind in kinds:
                if kind == "pot":
                    X = c.cur + int(0.9 * (P + c.owed))
                else:
                    X = my + c.my_bet
                X = int(min(X, c.opp_stack + c.opp_bet, my + c.my_bet))
                X = max(X, c.min_to)
                add = X - c.my_bet
                their_add = max(0, min(X - c.cur, opp_rem))
                jam = add >= my or their_add >= opp_rem
                f, rr, d = MODEL.vs_raise(st, (X - c.cur) / max(1.0, P + c.owed))
                eqc = eq_of(sd, wa * lk_call(q, min(0.97, f + d), rr, kap))
                f = max(0.0, f - d)
                if (jam and c.eff_bb > 40 and add > 3 * (P + c.owed)
                        and eqc - c.premium(add) < DEEP_JAM_EQ):
                    continue
                rx = 1.0 if (jam or st == "river") else R
                ev_called = rx * (eqc - c.premium(add)) * (P + add + their_add) - add
                ev = f * P + (1 - f) * ev_called + bonus
                opts.append(("raise", ev, X))
    opts.sort(key=lambda o: -o[1])
    best = opts[0]
    if len(opts) > 1 and opts[1][1] > best[1] - 0.025 * P and random.random() < 0.3:
        best = opts[1]
    dbg["opts"] = [(o[0], o[2], round(o[1] / c.bb, 2)) for o in opts[:4]]
    kind = best[0]
    if kind == "check":
        return {"action": "check"}
    if kind == "fold":
        return c.fold()
    if kind == "call":
        return c.check_call()
    return c.raise_to(c.cur + best[2] if kind == "bet" else best[2])


# =============================================================================
# Entry point
# =============================================================================


def _measure_speed():
    """Sustained evaluator speed. Runs ~0.3 s so CPU throttling in the sandbox (half a
    core, enforced in 100 ms windows) is included in the measurement."""
    rng = np.random.RandomState(7)
    arr = np.argsort(rng.random_sample((6000, 52)), axis=1)[:, :7]
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:
        hand_values(arr)
        n += 6000
    dt = max(1e-4, time.perf_counter() - t0)
    _SPEED["evals_per_s"] = min(3.0e6, max(1.0e5, n / dt))


def _fallback(gs):
    try:
        if gs.get("can_check"):
            return {"action": "check"}
        owed = gs.get("amount_owed") or 0
        pot = gs.get("pot") or 0
        r = [c[0] for c in gs.get("your_cards") or ["2c", "3d"]]
        strong = r[0] == r[1] or ("A" in r and ("K" in r or "Q" in r))
        if owed <= 0.25 * max(1, pot) or strong:
            return {"action": "call"}
    except Exception:
        pass
    return {"action": "fold"}


def decide(game_state):
    if game_state.get("type") == "warmup":
        try:
            _measure_speed()
        except Exception:
            pass
        return {"action": "check"}
    t0 = time.perf_counter()
    dbg = {}
    try:
        c = Ctx(game_state)
        hid = game_state.get("hand_id")
        if _CACHE["hand"] != hid:
            _CACHE["hand"] = hid
            _CACHE["streets"] = {}
        h = game_state.get("hand_num", 0)
        MODEL.update(game_state, c.opp)
        if c.street == "preflop":
            MODEL.depth.setdefault(h, c.eff_bb)
            MODEL.last_depth = MODEL.depth[h]
        else:
            MODEL.boards.setdefault(h, {})[c.street] = list(c.board)
        events, _, sb_id, fin = walk(game_state.get("action_log") or [])
        c.n_aggr = fin["n_aggr"] if fin["street"] == c.street else 0     # raises so far this street
        opp_sb = sb_id == c.opp
        w, sd = opp_range(c, events, opp_sb)
        if c.street == "preflop":
            out = preflop(c, w, fin["level"], dbg)
        else:
            if sd is None:
                sd = street_data(c.our, c.board)
            out = postflop(c, w, sd, opp_sb, dbg)
        fl = [k for k, v in c.flags.items() if v and k != "weak"]
        if fl:
            dbg["flags"] = fl
    except Exception as e:  # never crash: a crash is a fold
        dbg["error"] = repr(e)[:200]
        out = _fallback(game_state)
    if out.get("action") == "fold" and game_state.get("can_check"):
        out = {"action": "check"}
    elif out.get("action") == "check" and not game_state.get("can_check"):
        # never let a "check" turn into an unintended call
        owed = game_state.get("amount_owed") or 0
        out = {"action": "call" if owed <= 0.1 * max(1, game_state.get("pot") or 0) else "fold"}
    dbg["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["debug"] = dbg
    return out