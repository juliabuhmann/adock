# REPORT.md fact-check (fresh agent, 2026-10-27 ~10:05; transcribed by the PI because the agent could not write files)

Top errors (all fixed in the following commit):
1. Summary overclaimed "same or better accuracy than every classical baseline": v2_default (0.22 s) 52.6 % < Vina 55.4 / smina 55.1 / gnina no-CNN 59.3; v2_fitted 58.6 < gnina no-CNN 59.3 (NI not significant after Holm). Speed ratios are 46–54× vs GNINA, 12–14× vs Uni-Dock; C2 failed as a joint claim.
2. "Faster on 50/50": 47/50 vs Uni-Dock (3 instant failures); max Wilcoxon p 1.54e-12.
3. C7 reported seed 0 only; A6 asks for seeds 0–2 + McNemar. Seed means +3.5 / +3.0 / +2.0; PB2 per seed +4.9 / +1.1 / +4.6; "+5" lacked the post-freeze label; hetero-dimer +12.9 is the one favourable stratum (PB2 hetero n=12: −33).
4. C5 table "ADock alone" on v4 mixed the standalone run (11.2) with the in-hybrid run (11.8).
5. GNINA seeds 0–2 only on PB2 holo and RnP post-2023; seed 0 on PB-AF and dimers — undisclosed deviation from A1.
6. PB-AF hybrid row showed only T1; T3 (headline) 17.5 / 16.8 is below GNINA 20.9 / 18.3.
7. "No accuracy loss vs Vina" → "no significant difference (p ≥ 0.36)"; below Vina on DockGen/Astex val.
8. A6 registered after frozen ADock/hybrid outputs, not after all outputs (sealed baselines ran until 09:34).
9. PB-AF failures: 8 prep failures + 4 without AFDB model (12 AFDB, 8 ESMFold), not 4.
Minor: "≤ 0.01 pt" is for point estimates; gnina_flex 85/212 failures are timeouts (not "mostly"); template-search medians 3.4–3.6 / 0.7–0.9 s; interruption time source; C8 ranges pooled over seeds; C4 logit not reported; C6 RnP ADock-alone 49.5 missing; Uni-Dock 27/276 test failures not mentioned.
