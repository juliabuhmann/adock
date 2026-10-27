"""Power / CI-width estimates for DESIGN.md v0 claims (round-1 stats review).
Scenario values (success-rate differences, discordant fractions) are ASSUMPTIONS, not data.
Sample sizes come from results/data/*_counts.csv. Seeded Monte Carlo; rerun to reproduce."""
import numpy as np
from scipy.stats import binomtest
SEED, N_SIM, ALPHA = 0, 2000, 0.05
rng = np.random.default_rng(SEED)
# n per test stratum (from results/data/posebusters_counts.csv, runsnposes_counts.csv)
SIZES = {"PBv2_AF": 290, "PBv2_post2021": 214, "RnP_pred": 643, "RnP_pred_postBoltz2": 223,
         "RnP_hetero": 78, "PB_homo": 48, "PB_hetero": 14}
# (true paired difference, discordant fraction) -- assumed scenarios
SCEN = [(0.05, 0.15), (0.08, 0.20), (0.10, 0.20), (0.15, 0.25)]
def mcnemar_power(n, d, disc):
    p10, p01 = (disc + d) / 2, (disc - d) / 2
    x = rng.multinomial(n, [p10, p01, 1 - disc], size=N_SIM)
    hits = sum(1 for b, c, _ in x if b + c and b > c and binomtest(b, b + c, 0.5).pvalue < ALPHA)
    return hits / N_SIM
print("set,n,diff,disc,power_exact_mcnemar,paired_diff_95CI_halfwidth")
for name, n in SIZES.items():
    for d, disc in SCEN:
        print(f"{name},{n},{d},{disc},{mcnemar_power(n, d, disc):.2f},{1.96*np.sqrt((disc-d*d)/n):.3f}")
print("set,n,p,single_prop_95CI_halfwidth")
for name, n in SIZES.items():
    print(f"{name},{n},0.5,{1.96*np.sqrt(0.25/n):.3f}")
