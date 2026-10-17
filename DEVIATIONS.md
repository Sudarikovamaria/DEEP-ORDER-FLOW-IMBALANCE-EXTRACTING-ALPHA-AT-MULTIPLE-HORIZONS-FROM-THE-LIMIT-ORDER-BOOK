# Deviations from the paper

Kolm, Turiel, Westray (2023), *Deep Order Flow Imbalance*.

| # | What | Why | Effect |
|---|---|---|---|
| 1 | IEX DEEP data (L2, displayed liquidity only) instead of LOBSTER/Nasdaq | LOBSTER and Databento were not available | R²_OS lower than in the paper; the per-stock comparison of Section 4.2 is not directly comparable |
| 2 | `max_samples_per_epoch`: each epoch draws a random subset of 400k training windows | Training on a MacBook M1 Pro | Fewer optimizer steps per epoch; heavier models may be undertrained (see 13) |
| 3 | 11 stocks (6 for the neural networks, see 11) × 12 weeks (3 rolling windows) instead of 115 stocks × 48 weeks (18 windows) | Compute budget and data download | Less statistical power in Tables 3–5 (15 test days per stock) |
| 4 | `of_variant` defaults to `paper` (formulas (2)–(3) as printed); the Cont et al. (2014) variant is available in the config | When a level moves away the paper uses −v(t), Cont et al. use −v(t−1) | Only the `paper` variant was run within the deadline |
| 5 | Composite events are collapsed by the DEEP event flag (a snapshot only after a price level update with flag 0x1), not by identical timestamps (Appendix A) | DEEP marks the end of an atomic book update explicitly | Practically equivalent; in-transition book states never enter the data |
| 6 | A day with a trading halt inside 09:40–15:50 is dropped entirely (`halt_policy: drop_day`) | Targets and windows across a halt are not meaningful; the paper excludes stocks with corporate actions and says nothing about halts | No halts occurred for the selected stocks in the period |
| 7 | ARX is estimated by closed-form ridge regression (unpenalized intercept, λ chosen by validation MSE) instead of Adam with lr 1e-4 and early stopping | Fast and deterministic, no dependence on learning rate or epoch count; early stopping and ridge are related forms of regularization | Checked with `--sgd` (Adam, as in the paper): ridge is stronger on OF (0.27% vs −0.25%) and OFI (0.28% vs 0.05%), and removes the OFI-over-OF gain of Appendix B.1 |
| 8 | Stocks are selected by a rule: common stock, Δt ≥ 180 ms, ≥ 300k book events per day, on the day before the period (20260717); the paper uses Nasdaq 100 members | The displayed IEX book flickers, so Δt is 3–70 times smaller than for the same stocks on Nasdaq; with Δt < 50 ms the first horizon h₁ = Δt/5 is shorter than the δτ = 10 ms latency. 180 ms is the smallest Δt among the paper's stocks (AAPL, Table 6) | The sample leans towards large-tick stocks; less spread in tick size for Table 5 |
| 9 | The LOB input is run only for ARX (trained with Adam) and MLP; main runs use OF, and ARX is also run on OFI (Appendix B.1 check) | Compute budget: on a MacBook M1 Pro, ridge ARX on LOB (4000 features) would take ~10 h and CNN-LSTM on LOB ~30 h for 11 stocks; renting a GPU was not planned | OF vs LOB (Table 3) only for ARX and MLP; no LOB rows for the LSTM-based models in Figures 2, 3 and Table 4 |
| 10 | Early stopping of the neural networks monitors a fixed random subset of the validation week (`max_val_samples` = 200k windows, the same on every epoch) instead of all ~1.5–2M windows; the test week is always complete, and ARX uses the full validation week | On the M1 a full validation pass after every epoch takes longer than the training epoch itself | Epochs about twice as fast; the stopping criterion is slightly noisier (~3× larger standard error); test metrics are unaffected |
| 11 | The neural networks (MLP, LSTM, LSTM (3), LSTM-MLP, CNN-LSTM) are trained on 6 stocks: BAC, VZ, NKE, USB, OXY, UBER; ARX on all 11 | One week for the whole project and limited time on the team's laptops. Selected by tick size, which is known before training: every stock with a one-tick spread ≥ 70% of the time, plus UBER as a small-tick control | Model comparisons use the 6 common stocks; Table 5 has few small-tick stocks, so the tick-size conclusions rest mostly on ARX over 11 stocks |
| 12 | LSTM (3) is trained for at most 12 epochs (`--epochs 12`) instead of 50 | At learning rate 1e-4 one epoch of the three-layer LSTM takes ~3 min on the M1; 50 epochs on 6 stocks did not fit the ~24 h deadline | The cap was reached in 6 of 18 windows (early stopping ended the other 12); LSTM (3) may be slightly undertrained, so its R²_OS is biased down rather than up |
| 13 | Models with learning rate 1e-5 in Table 2 (LSTM, LSTM (3), LSTM-MLP, MLP) are trained with 1e-4 (`--learning-rate 1e-4`) | Our epoch (400k windows) is ~18 times shorter than a full pass over the training set, for which 1e-5 was chosen. Check `scripts/lr_check.py` (LSTM, BAC, window 1, 4 epochs): best validation MSE 1.1821 at 1e-4 vs 1.1845 at 1e-5, and 1e-4 is ahead after the first epoch; chosen on validation only | The models train within our step budget; for MLP and LSTM-MLP the choice is carried over from the LSTM check without a separate test |

## Decisions where the paper is silent

| Question | Decision |
|---|---|
| Period for N in Δt = 2.34·10⁷ / N | Past data only: the training weeks of each window (the paper does not say; probably the whole sample) |
| R²_OS benchmark ("average return on the test set") | The mean return of the test week; R²_OS is computed for each day of that week |
| Empty book levels | OF: an empty level compares as −∞ (bid) / +∞ (ask); LOB: the price of the deepest non-empty level on that side, volume 0 |
| Order of OF columns before block 2 of the CNN-LSTM | Columns are interleaved as (bOF_i, aOF_i) pairs so the 1×2 convolution combines both sides of a level, as Table 1 describes |
| Long-horizon grid (Figure 8) | 0.2…2.0 in steps of 0.2 and 3…10 in steps of 1 (in Δt); the long targets are built but Figure 8 was not run |
| Test metric | R²_OS on the raw (not winsorized) test targets; forecasts are mapped back from z-scores |
