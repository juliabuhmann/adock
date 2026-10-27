# REPORT.md fact-check 2 (fresh agent, 2026-10-27 ~12:45; transcribed by the PI) — sections added after fact-check 1
Errors (fixed in the following commit):
1. §6.6 table: ADock v2_fitted AFDB "all 222" was the in-hybrid run (12.2); the seed-0 standalone value used for C9 is 10.8.
2. "Co-folding" paragraph: "only receptor preparation that improved physics docking" contradicted by pocket-transfer ensemble (+5.6 [1.9, 10.3] confirm) → reworded to "largest receptor-side gain ... (+8.3 pooled; confirm +6.0, ns)".
3. §6.5: CPU parallel efficiency 0.94–0.99 up to 16 threads (not 0.96).
RnP similarity reference: supported only indirectly (script docstring, LITERATURE.md); wording now says "per the RnP paper ... not re-verified for the Zenodo v2026-01 annotations".
Minor (fixed): pooled val sets noted for pocket-transfer/oracle rows; "backbone alone +6–7" marked ns; stale "pre-registered test" wording in §8; stray punctuation; single-driver throughput source; 80→81 %; reports/c9_cofold_test.md caveat updated with the similarity analysis.
Not changed: C1 low-sim bin n=21 (v4 subset) vs C9 n=22 (all 222) — different universes.
