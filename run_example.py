"""Standalone demo of the block that replaces `# Call the random match here`."""

import point_iid_dp as tdp


# ---- stand-ins so this file runs on its own; in your script these already exist ----
def _fake_row():
    class Row(dict):
        def __getitem__(self, k):
            return dict.__getitem__(self, k)

    return Row(
        best_of=5,
        tourney_name="Roland Garros",
        round="R32",
        score="6-3 6-4 7-6(4)",
        winner_rank=4,
        loser_rank=71,
    )


example = _fake_row()
players = [[0.66, 0, 0, 0, 0, "Alcaraz C."], [0.61, 0, 0, 0, 0, "Struff J."]]
pA_serve, pB_serve = 0.66, 0.61
# ------------------------------------------------------------------------------------

name_a, name_b = players[0][5], players[1][5]
best_of = int(example["best_of"])

# 1) exact joint distribution over match outcomes, averaged over the serve coin toss
joint = tdp.match_dist_coin_toss(
    pA_serve,
    pB_serve,
    best_of=best_of,
    rules="tb7",  # set per-tournament; adv/tb10/tb7_at_12 for older majors
)

# 2) read markets straight off the joint -- each is an exact tail sum, no sampling
p_a = tdp.p_match(joint, best_of)
p_b = 1.0 - p_a

games_d = tdp.total_games_dist(joint)
spread_d = tdp.games_spread_dist(joint)
sets_d = tdp.set_betting_dist(joint)
tb_d = tdp.tiebreak_count_dist(joint)

print(f"=== {name_a} vs {name_b}  (best of {best_of}) ===\n")
print(f"Match winner:")
print(f"  {name_a:<12} {p_a:6.3f}   {tdp.calc_odds(p_a):}")
print(f"  {name_b:<12} {p_b:6.3f}   {tdp.calc_odds(p_b):}\n")

print(f"Total games:  E = {tdp.expected(games_d):.2f}")
for line in (int(tdp.expected(games_d)) - 2.5, int(tdp.expected(games_d)) + 0.5):
    over = tdp.p_over(games_d, line)
    print(
        f"  over {line:4.1f}  {over:6.3f} ({tdp.calc_odds(over):})   "
        f"under {line:4.1f}  {1-over:6.3f} ({tdp.calc_odds(1-over):})"
    )

print(f"\nSet betting (most likely):")
for sc, p in sorted(sets_d.items(), key=lambda x: -x[1])[:4]:
    print(f"  {sc[0]}-{sc[1]}   {p:6.3f}   {tdp.calc_odds(p):}")

print(f"\nTiebreaks:  E = {tdp.expected(tb_d):.2f}")
print(
    f"  at least one   {tdp.p_over(tb_d, 0.5):6.3f} ({tdp.calc_odds(tdp.p_over(tb_d,0.5)):})"
)
