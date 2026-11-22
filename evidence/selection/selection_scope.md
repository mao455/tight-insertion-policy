The released A_130_50 is the highest observed full autonomous result among four
already known frozen24 comparisons, not an independent estimate after selection.
All 96 attempts were legal and software-complete. Neither a takeover rescue,
training-family success, first-subgoal completion, validation loss alone, nor a
chosen successful GUI seed was used as the published success-rate denominator.

The exact pair was originally selected on validation6 among checkpoints meeting
at least 50 new visits per training episode and 50 distinct optimizer-step
exposures per legal expert tick. `training_completion.csv` records A DP epoch51 /
update4845 and matching rear epoch50 / update16174 (both actually trained60epochs).
Parents were the selected72 A pair, with fresh AdamW and new DP EMA for this
continuation. `configs/training130.json` fixes all 130 train + 6 validation IDs,
raw hashes and family membership. Release parents permit reproduction of this
stage without requiring the separate historical 400-episode archive.

The 2026-11-19 same-wave development and training-family evaluation is retained
separately in `per_case.csv` and `counts.csv`: A_130_50 development3/8, training2/8;
B_130_50 development2/8, training2/8. These reused development starts are not new
test observations. Other historical test sets, including the selected72 A/B
3/20 each, have different initial states and are not pooled with frozen24.

The frozen24 registration predates these results, uses two distinct cohorts
(reserved8 and fresh16), and hashes the complete DP+Transformer pairs. Native
120Hz physics, 60Hz control, 5Hz replanning, full causal sensor histories, DDIM
sampler7091 and strict complete-insertion evaluation are retained. A predicts
actual reached joint configurations with offline geometry supervision; B predicts
expert commands. Geometry labels and expert access are absent from the GUI flow.

The later alignment pilot used takeover diagnostics and different starts. The
separate paired-DAgger run was still reconstructing parents at the course audit;
no stronger completed comparable autonomous result was available. This is
rechecked at final publication, without modifying or stopping that research run.

Course GUI seed28853000 was fixed before any GUI execution. The startup OOM and
subsequent valid timeout are retained as acceptance evidence and excluded from
the historical5/24. No initial-state substitution or search for success occurred.
