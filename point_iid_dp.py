# Final set rule configurations
from collections import defaultdict
from functools import lru_cache

import pandas as pd

RULES = {
    "tb7": dict(tb_at=6, tb_target=7),  # standard
    "tb10": dict(tb_at=6, tb_target=10),  # current majors' final set
    "adv": dict(tb_at=None, tb_target=7),  # advantage set (pre-2019 AO/RG/W/-USO)
    "tb7_at_12": dict(tb_at=12, tb_target=7),  # Wimbledon 2019-2021 final set
}


# Given p: the probability of a server winning a point, use the negative binomial distribution to calculate the percentage
# the server wins the game in addition to the geometric distribution for deuces
def prob_hold(p: float) -> float:
    if p <= 0:
        return 0
    if p >= 1:
        return 1

    q = 1 - p
    # deuce (See notes for derivation)
    deuce = p * p / (p * p + q * q)
    # (40-0) + (40-15) + (40-30) + deuce (Negative Binomial)
    hold = p**4 + 4 * p**4 * q + 10 * p**4 * q**2 + 20 * p**3 * q**3 * deuce

    return hold


# Calculate the games distribution from the same formulas as above and return a dictionary
# d[(server won T/F, number of games)]
# Game can end at 4, 5, 6 or reaches deuce where it can end at 8, 10, 12, ...
def game_length_dist(p, max_deuce_cycles=40):
    q = 1.0 - p
    d = {}
    for pts_lost, ways in ((0, 1), (1, 4), (2, 10)):
        d[(True, 4 + pts_lost)] = ways * p**4 * q**pts_lost
        d[(False, 4 + pts_lost)] = ways * q**4 * p**pts_lost
    reach_deuce = 20.0 * p**3 * q**3
    for m in range(max_deuce_cycles):
        base = reach_deuce * (2.0 * p * q) ** m
        d[(True, 8 + 2 * m)] = d.get((True, 8 + 2 * m), 0.0) + base * p * p
        d[(False, 8 + 2 * m)] = d.get((False, 8 + 2 * m), 0.0) + base * q * q
    return d


# Determines the server of the current tiebreak
# Starts at t=0, first server goes then alternates every two
def server_of_point(t, first="A"):
    other = "B" if first == "A" else "A"
    return first if ((t + 1) // 2) % 2 == 0 else other


# Calculating the chance a server's (determined by first) probability of winning a tiebreak
# At each score, calculates next score by either multiplying by pA if A wins or pB if B wins
# Accumulates total probability of leaf nodes where player A has won and caches the results
#
# NOTE To avoid neverending loops, from_level uses conditional probability to calculate in closed-form A's chances of winning from 6-6 9-9
@lru_cache(maxsize=None)
def tiebreak_win_prob(pA, pB, first="A", target=7):
    """P(A wins the tiebreak). Exact; closed form at (target-1, target-1)."""
    num = pA * (1.0 - pB)
    den = num + (1.0 - pA) * pB
    from_level = num / den if den > 0 else 0.5  # every 2-pt block = one serve each

    winA = 0.0
    dist = {(0, 0): 1.0}
    while dist:
        nxt = defaultdict(float)
        for (a, b), pr in dist.items():
            pa = pA if server_of_point(a + b, first) == "A" else 1.0 - pB
            for da, w in ((1, pa), (0, 1.0 - pa)):
                na, nb = a + da, b + (1 - da)
                p2 = pr * w
                if na >= target and na - nb >= 2:
                    winA += p2
                elif nb >= target and nb - na >= 2:
                    pass
                elif na == target - 1 and nb == target - 1:
                    winA += p2 * from_level
                else:
                    nxt[(na, nb)] += p2
        dist = nxt
    return winA


@lru_cache(maxsize=None)
def set_dist(pA, pB, first="A", tb_at=6, tb_target=7, max_pairs=200):
    """
    Full joint over set outcomes: {(games_A, games_B, tiebreak_played): prob}.

    `first` is the server of the set's first game. Games alternate, so the server
    of game k is A iff (k % 2 == 0) == (first == 'A').

    tb_at=None (advantage set) has an unbounded score, so 6-6 is collapsed with
    the same pair-geometric trick used for deuce -- exact, not truncated.
    """
    hA, hB = prob_hold(pA), prob_hold(pB)
    out = defaultdict(float)
    dist = {(0, 0): 1.0}

    while dist:
        nxt = defaultdict(float)
        for (ga, gb), pr in dist.items():
            a_serves = ((ga + gb) % 2 == 0) == (first == "A")
            pa = hA if a_serves else 1.0 - hB  # P(A wins this game)
            for da, w in ((1, pa), (0, 1.0 - pa)):
                nga, ngb = ga + da, gb + (1 - da)
                p2 = pr * w
                if max(nga, ngb) >= 6 and abs(nga - ngb) >= 2:
                    out[(nga, ngb, False)] += p2
                elif tb_at is not None and nga == tb_at and ngb == tb_at:
                    # game index 2*tb_at is even -> served by the set's first server
                    wa = tiebreak_win_prob(pA, pB, first, tb_target)
                    out[(nga + 1, ngb, True)] += p2 * wa
                    out[(nga, ngb + 1, True)] += p2 * (1.0 - wa)
                elif tb_at is None and nga == 6 and ngb == 6:
                    _advantage_tail(out, p2, hA, hB, max_pairs)
                else:
                    nxt[(nga, ngb)] += p2
        dist = nxt
    return tuple(sorted(out.items()))  # hashable, for lru_cache


def _advantage_tail(out, mass, hA, hB, max_pairs):
    """
    Advantage set from 6-6. Games come in pairs (one serve each), so the process
    is memoryless: A takes the pair w.p. hA*(1-hB), B w.p. (1-hA)*hB, else 6-6
    repeats one game higher. Terminal scores are (8+m, 6+m) / (6+m, 8+m).

    NOTE: a naive "stop at max_games" truncation is WRONG here -- it produces
    tied game scores, which the match DP reads as neither player winning the set,
    and the sweep never terminates.
    """
    Pa = hA * (1.0 - hB)
    Pb = (1.0 - hA) * hB
    split = 1.0 - Pa - Pb
    if Pa + Pb <= 0.0:  # degenerate: both always hold
        out[(7, 7, False)] += mass
        return
    for m in range(max_pairs):
        s = split**m
        out[(8 + m, 6 + m, False)] += mass * s * Pa
        out[(6 + m, 8 + m, False)] += mass * s * Pb
    tail = split**max_pairs  # lump the residue, preserve mass
    m = max_pairs - 1
    out[(8 + m, 6 + m, False)] += mass * tail * Pa / (Pa + Pb)
    out[(6 + m, 8 + m, False)] += mass * tail * Pb / (Pa + Pb)


def match_dist(pA, pB, best_of=3, first="A", rules="tb7", final_rules=None):
    """
    Exact joint over match outcomes:
        {(sets_A, sets_B, games_A, games_B, n_tiebreaks): prob}

    `rules` / `final_rules` are keys of RULES. `final_rules` defaults to `rules`
    and applies only to the deciding set.
    """
    normal = RULES[rules]
    final = RULES[final_rules or rules]
    to_win = best_of // 2 + 1

    # Create a cache of set distributions = (first server, rules, normal or final set) = (games won A, games won B, tiebreak 1/0)
    # As over an entire match, the only things that can differ is if the first server is A or B and if it is a normal set or the final set
    # Return the set distribution
    cache = {}

    def sd(f, cfg):
        key = (f, cfg["tb_at"], cfg["tb_target"])
        if key not in cache:
            cache[key] = set_dist(pA, pB, f, cfg["tb_at"], cfg["tb_target"])
        return cache[key]

    out = defaultdict(float)
    states = {(0, 0, first, 0, 0, 0): 1.0}

    while states:
        nxt = defaultdict(float)
        for (sa, sb, srv, ca, cb, ntb), pr in states.items():
            is_final = (sa == to_win - 1) and (sb == to_win - 1)
            for (ga, gb, tb), q in sd(srv, final if is_final else normal):
                nsa, nsb = sa + (ga > gb), sb + (gb > ga)
                nca, ncb = ca + ga, cb + gb
                nntb = ntb + (1 if tb else 0)
                # serve alternates: next set's first server flips iff the set had
                # an odd number of games
                nsrv = srv if (ga + gb) % 2 == 0 else ("B" if srv == "A" else "A")
                p2 = pr * q
                if nsa == to_win or nsb == to_win:
                    out[(nsa, nsb, nca, ncb, nntb)] += p2
                else:
                    nxt[(nsa, nsb, nsrv, nca, ncb, nntb)] += p2
        states = nxt
    return dict(out)


# Use when the first server is unknown, in seeded tournaments the higher seed gets the first serve
def match_dist_coin_toss(pA, pB, **kw):
    """Average over the coin toss when the first server is unknown."""
    kw.pop("first", None)
    a = match_dist(pA, pB, first="A", **kw)
    b = match_dist(pA, pB, first="B", **kw)
    out = defaultdict(float)
    for d in (a, b):
        for k, v in d.items():
            out[k] += 0.5 * v
    return dict(out)


def posterior_predictive(draws, best_of=3, rules="tb7", final_rules=None):
    """
    Average the joint over posterior draws [(pA, pB), ...] instead of plugging in
    point estimates. E[f(p)] != f(E[p]) and the gap is largest exactly on the
    derivative markets.
    """
    out = defaultdict(float)
    w = 1.0 / len(draws)
    for pA, pB in draws:
        for k, v in match_dist_coin_toss(
            pA, pB, best_of=best_of, rules=rules, final_rules=final_rules
        ).items():
            out[k] += w * v
    return dict(out)


def tour_avg_return(df: pd.DataFrame):
    serve_pts_total = (df["w_svpt"] + df["l_svpt"]).sum()
    serve_pts_won = (
        df["w_1stWon"] + df["w_2ndWon"] + df["l_1stWon"] + df["l_2ndWon"]
    ).sum()
    tour_avg_return = 1 - serve_pts_won / serve_pts_total
    return tour_avg_return


def serve_win_prob(server_stats, returner_stats, tour_avg_return):
    """
    Barnett-Clarke: P(server wins point) = server's serve win % - opponent's
    return win % + tour-average return win %. Both raw stats already reflect
    average opposition, so this adjusts relative to the average instead of
    double-counting.
    Stats format: [%1st in, %1st won, %2nd won, %1st return won, %2nd return won, name]
    """

    fs, fs_win, ss_win = server_stats[:3]
    serve_pct = fs * fs_win + (1 - fs) * ss_win

    # Returner's overall return win %, weighted by the server's 1st-serve-in rate
    r_fs_win, r_ss_win = returner_stats[3:5]
    return_pct = fs * r_fs_win + (1 - fs) * r_ss_win

    return serve_pct - return_pct + tour_avg_return


def get_stats(match: pd.Series) -> list:
    cols = [
        "first_in_pct",
        "first_won_pct",
        "second_won_pct",
        "first_ret_won_pct",
        "second_ret_won_pct",
    ]
    return [
        [float(match[f"w_prior_{c}"]) for c in cols] + [match["winner_name"]],
        [float(match[f"l_prior_{c}"]) for c in cols] + [match["loser_name"]],
    ]


# --------------------------------------------------------------------------
# Markets: every one of these is an exact tail sum over the joint
# --------------------------------------------------------------------------
def p_match(joint, best_of=3):
    to_win = best_of // 2 + 1
    return sum(v for (sa, _, _, _, _), v in joint.items() if sa == to_win)


def total_games_dist(joint):
    d = defaultdict(float)
    for (_, _, ca, cb, _), v in joint.items():
        d[ca + cb] += v
    return dict(d)


def games_spread_dist(joint):
    d = defaultdict(float)
    for (_, _, ca, cb, _), v in joint.items():
        d[ca - cb] += v
    return dict(d)


def set_betting_dist(joint):
    d = defaultdict(float)
    for (sa, sb, _, _, _), v in joint.items():
        d[(sa, sb)] += v
    return dict(d)


def tiebreak_count_dist(joint):
    d = defaultdict(float)
    for (_, _, _, _, ntb), v in joint.items():
        d[ntb] += v
    return dict(d)


def p_over(dist, line):
    """P(X > line) for a half-point line."""
    return sum(v for k, v in dist.items() if k > line)


def calc_odds(p, format="euro"):
    if format == "euro":
        return round(1 / p, 2)
    if format == "american":
        if p <= 0 or p >= 1:
            return float("inf") if p <= 0 else float("-inf")
        return round((1 - p) * 100 / p) if p < 0.5 else round(-p * 100 / (1 - p))


def expected(dist):
    return sum(k * v for k, v in dist.items())


# --------------------------------------------------------------------------
# Posterior integration: the reason the DP exists
# --------------------------------------------------------------------------
def posterior_predictive(draws, best_of=3, rules="tb7", final_rules=None):
    """
    Average the joint over posterior draws [(pA, pB), ...] instead of plugging in
    point estimates. E[f(p)] != f(E[p]) and the gap is largest exactly on the
    derivative markets.
    """
    out = defaultdict(float)
    w = 1.0 / len(draws)
    for pA, pB in draws:
        for k, v in match_dist_coin_toss(
            pA, pB, best_of=best_of, rules=rules, final_rules=final_rules
        ).items():
            out[k] += w * v
    return dict(out)


if __name__ == "__main__":
    match_features = pd.read_csv("./features/match_features.csv")
    majors = match_features[match_features["tourney_level"] == "G"]
    ta = tour_avg_return(majors)

    examples = majors.sample(5)

    for i in range(len(examples)):
        example = examples.iloc[i]

        players = get_stats(example)

        pA_serve = serve_win_prob(players[0], players[1], ta)
        pB_serve = serve_win_prob(players[1], players[0], ta)

        print(f"{example["tourney_name"]} - {str(example["tourney_date"])[:4]}")

        print(
            f"P(point win on serve): {players[0][5]} = {pA_serve:.3f}, {players[1][5]} = {pB_serve:.3f}"
        )

        # --- price the match off the exact DP joint ---
        name_a, name_b = players[0][5], players[1][5]
        best_of = int(example["best_of"])

        joint = match_dist_coin_toss(
            pA_serve,
            pB_serve,
            best_of=best_of,
            rules="tb7",  # set per tournament; see note below
        )

        p_a = p_match(joint, best_of)
        games_d = total_games_dist(joint)
        spread_d = games_spread_dist(joint)
        sets_d = set_betting_dist(joint)
        tb_d = tiebreak_count_dist(joint)

        print(
            f"{name_a}: {p_a:.3f} ({calc_odds(p_a)}) | "
            f"{name_b}: {1-p_a:.3f} ({calc_odds(1-p_a)})"
        )
        print(
            f"E[games] = {expected(games_d):.2f}, "
            f"P(>=1 TB) = {p_over(tb_d, 0.5):.3f}\n"
        )
