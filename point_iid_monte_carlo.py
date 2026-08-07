import math
import random

import numpy as np
import pandas as pd

VERBOSE = False  # set True to print each match's scoreboard

# ATP tour averages (adjust per season/surface if you have them)
TOUR_AVG_RETURN = 0.365  # average % of return points won on tour


def simulate_game(server, pA_serve, pB_serve):
    """Play one game. Returns 'A' or 'B'. Handles deuce implicitly."""
    serve_pct = pA_serve if server == "A" else pB_serve
    receiver = "B" if server == "A" else "A"
    pts = {server: 0, receiver: 0}

    while True:
        winner = server if random.random() < serve_pct else receiver
        pts[winner] += 1
        if (
            pts[winner] >= 4
            and pts[winner] - pts[receiver if winner == server else server] >= 2
        ):
            return winner


def simulate_tiebreak(server, pA_serve, pB_serve, target=7):
    """First to `target`, win by 2. Server serves 1 point, then alternate every 2."""
    pts = {"A": 0, "B": 0}
    point_num = 0

    while True:
        serve_pct = pA_serve if server == "A" else pB_serve
        receiver = "B" if server == "A" else "A"
        winner = server if random.random() < serve_pct else receiver
        pts[winner] += 1
        point_num += 1

        if (
            pts[winner] >= target
            and pts[winner] - pts[receiver if winner == server else server] >= 2
        ):
            return winner

        # switch server after point 1, then every 2 points
        if point_num % 2 == 1:
            server = receiver


def simulate_set(server, pA_serve, pB_serve):
    """Returns (set_winner, next_server, games_A, games_B)."""
    games = {"A": 0, "B": 0}

    while True:
        winner = simulate_game(server, pA_serve, pB_serve)
        games[winner] += 1
        server = "B" if server == "A" else "A"  # server now = next game's server

        if (
            games[winner] >= 6
            and games[winner] - games["B" if winner == "A" else "A"] >= 2
        ):
            return winner, server, games["A"], games["B"]

        if games["A"] == 6 and games["B"] == 6:
            tb_winner = simulate_tiebreak(server, pA_serve, pB_serve)
            games[tb_winner] += 1
            # player who received first in the tiebreak serves first next set
            next_server = "B" if server == "A" else "A"
            return tb_winner, next_server, games["A"], games["B"]


def simulate_match(pA_serve, pB_serve, best_of=5):
    """Returns (match_winner, scoreboard) where scoreboard is a list of (games_A, games_B)."""
    sets = {"A": 0, "B": 0}
    scoreboard = []
    server = random.choice(["A", "B"])
    sets_to_win = best_of // 2 + 1

    while sets["A"] < sets_to_win and sets["B"] < sets_to_win:
        set_winner, server, games_A, games_B = simulate_set(server, pA_serve, pB_serve)
        sets[set_winner] += 1
        scoreboard.append((games_A, games_B))

    return ("A" if sets["A"] == sets_to_win else "B"), scoreboard


def print_scoreboard(scoreboard, name_a, name_b):
    width = max(len(name_a), len(name_b)) + 2
    header = " " * width + "".join(f"Set {i+1}  " for i in range(len(scoreboard)))
    print(header)
    print("-" * len(header))
    print(name_a.ljust(width) + "".join(f"{a:^7}" for a, _ in scoreboard))
    print(name_b.ljust(width) + "".join(f"{b:^7}" for _, b in scoreboard))
    print()


def calc_odds(p):
    """American odds from a win probability (0 < p < 1)."""
    if p <= 0 or p >= 1:
        return float("inf") if p <= 0 else float("-inf")
    return round((1 - p) * 100 / p) if p < 0.5 else round(-p * 100 / (1 - p))


def serve_win_prob(server_stats, returner_stats, tour_avg_return=TOUR_AVG_RETURN):
    """
    Barnett-Clarke: P(server wins point) = server's serve win % - opponent's
    return win % + tour-average return win %. Both raw stats already reflect
    average opposition, so this adjusts relative to the average instead of
    double-counting.
    Stats format: [%1st in, %1st won, %2nd won, %1st return won, %2nd return won, name]
    """
    df = pd.read_csv("./features/match_features.csv")
    serve_pts_total = (df["w_svpt"] + df["l_svpt"]).sum()
    serve_pts_won = (
        df["w_1stWon"] + df["w_2ndWon"] + df["l_1stWon"] + df["l_2ndWon"]
    ).sum()
    tour_avg_return = 1 - serve_pts_won / serve_pts_total

    fs, fs_win, ss_win = server_stats[:3]
    serve_pct = fs * fs_win + (1 - fs) * ss_win

    # Returner's overall return win %, weighted by the server's 1st-serve-in rate
    r_fs_win, r_ss_win = returner_stats[3:5]
    return_pct = fs * r_fs_win + (1 - fs) * r_ss_win

    return serve_pct - return_pct + tour_avg_return


def set_score_distribution(
    pA_serve,
    pB_serve,
    n_sets=100_000,
    name_a="Player A",
    name_b="Player B",
    show=True,
    save_path=None,
):
    """Simulate `n_sets` independent sets with user-chosen serve win %s and
    plot the frequency of each set score (e.g. 6-3, 7-6) as a probability.

    pA_serve / pB_serve: P(point win on own serve) for each player -- pass any
    values you like, no stat calculations needed.

    X axis: scores formatted "A-B" (Player A's games first).
    Y axis: probability = (# sets ending with that score) / n_sets.

    Returns a pandas Series of probabilities indexed by score string.
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    counts = Counter()
    for _ in range(n_sets):
        server = random.choice(["A", "B"])
        _, _, games_A, games_B = simulate_set(server, pA_serve, pB_serve)
        counts[(games_A, games_B)] += 1

    # Order scores by frequency, most common first
    ordered = [s for s, _ in counts.most_common()]

    labels = [f"{a}-{b}" for a, b in ordered]
    probs = [counts[s] / n_sets for s in ordered]
    result = pd.Series(probs, index=labels, name="probability")

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("summer")
    colors = cmap(np.linspace(0, 1, len(ordered)))
    ax.bar(labels, probs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel(f"Set score ({name_a} - {name_b})")
    ax.set_ylabel("Probability (sets / total sets)")
    ax.set_title(
        f"Set score distribution | {name_a} serve={pA_serve:.0%}, "
        f"{name_b} serve={pB_serve:.0%} | n={n_sets:,}"
    )
    for i, p in enumerate(probs):
        ax.text(i, p, f"{p:.1%}", ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return result


def monte_simulation(
    pA_serve,
    pB_serve,
    n,
    best_of=5,
    name_a="Player A",
    name_b="Player B",
):
    wins_A = 0
    game_spread = []
    set_spread = []

    for _ in range(n):
        winner, scoreboard = simulate_match(pA_serve, pB_serve, best_of)

        # Gather main spreads
        game_diff = sum(a - b for a, b in scoreboard)
        set_diff = sum(1 if a > b else -1 for a, b in scoreboard)
        game_spread.append(game_diff)
        set_spread.append(set_diff)

        if winner == "A":
            wins_A += 1
        if VERBOSE:
            print_scoreboard(scoreboard, name_a, name_b)

    p = wins_A / n
    # 95% CI (normal approximation)
    margin = 1.96 * math.sqrt(p * (1 - p) / n)
    lo, hi = max(0.0, p - margin), min(1.0, p + margin)

    print(f"{name_a} win % = {p:.3f}  (95% CI: {lo:.3f} - {hi:.3f})")
    print(f"{name_b} win % = {1 - p:.3f}  (95% CI: {1 - hi:.3f} - {1 - lo:.3f})")
    print(f"{name_a} odds = {(1/p):.2f} || {name_b} odds = {(1/(1 - p)):.2f}")

    # Gather average outcomes from game and set distributions
    # If Player A is favored it will be -3.5 games (needs to win by 4) else it will be +3.5 games (needs to lose by less than 4)
    avg_games = round(np.mean(game_spread), 0)
    avg_sets = round(np.mean(set_spread), 0)
    a_cover_games = np.where(game_spread > avg_games + 0.5, 1, 0).sum() / len(
        game_spread
    )
    a_cover_sets = np.where(set_spread > avg_sets + 0.5, 1, 0).sum() / len(set_spread)

    print(
        f"{name_a} to cover {"+" if avg_games < 0 else "-"}{abs(avg_games + 0.5)} games: {100 * a_cover_games:.2f}% {(1/a_cover_games):.2f}"
    )
    print(
        f"{name_a} to cover {"+" if avg_sets < 0 else "-"}{abs(avg_sets + 0.5)} sets: {100 * a_cover_sets:.2f}% {(1/a_cover_sets):.2f}"
    )


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


if __name__ == "__main__":
    # [%1st serve in, %1st serve pts won, %2nd serve pts won,
    #  %1st serve RETURN pts won, %2nd serve RETURN pts won, name]
    match_features = pd.read_csv("./features/match_features.csv")

    majors = match_features[match_features["tourney_level"] == "G"]
    mismatches = match_features[
        (abs(match_features["winner_rank"] - match_features["loser_rank"]) >= 100)
        & match_features["winner_rank"]
        <= 50
    ]

    random_match = random.randint(0, len(mismatches) - 1)
    ex_match = mismatches.iloc[random_match, :]

    print(
        f"{ex_match["tourney_name"]} {ex_match["round"]} #{int(ex_match["winner_rank"])} {ex_match["winner_name"]} {ex_match["score"]} #{int(ex_match["loser_rank"])} {ex_match["loser_name"]}"
    )

    players = get_stats(ex_match)

    pA_serve = serve_win_prob(players[0], players[1])
    pB_serve = serve_win_prob(players[1], players[0])

    print(
        f"P(point win on serve): {players[0][5]} = {pA_serve:.3f}, {players[1][5]} = {pB_serve:.3f}\n"
    )

    best_of = int(ex_match["best_of"])

    monte_simulation(
        pA_serve,
        pB_serve,
        500_000,
        best_of,
        name_a=players[0][5],
        name_b=players[1][5],
    )

    # set_score_distribution(
    #     pA_serve=0.85,
    #     pB_serve=0.62,
    #     save_path="./graphs/set_dist_high_discrep_serve.png",
    # )
