# v0.9.1 — Factor Contribution Report

## Purpose

The Evolutionary Universe Laboratory identifies competitive factor combinations. The Factor Contribution Report explains why those combinations survived.

This version does not add predictors, market variables or data downloads. It analyzes the `Leaderboard` produced by v0.9 and answers six research questions:

1. Which factors appear most often among the strongest candidates?
2. How much does each factor add when introduced into an existing combination?
3. Which contextual factors strengthen or weaken that marginal contribution?
4. Which pairs perform better than their individual components?
5. At what factor count does performance begin to saturate?
6. Are additional factors increasing validation quality or only complexity?

## Command

```bat
market-factor-contribution ^
  --input reports\adaptive_universe\evolutionary_universe_v090_4f.xlsx ^
  --sheet Leaderboard ^
  --score-column evolution_score ^
  --top 25 ^
  --export ^
  --output reports\adaptive_universe\factor_contribution_v091.xlsx
```

The input filename can be changed to the actual v0.9 workbook.

## Output sheets

### Factor_Ranking

Ranks factors using:

- appearances in the selected Top N;
- rank-weighted presence;
- best and average evolution score;
- average validation Rare Event F1;
- average balanced accuracy;
- average generalization gap.

Frequent presence is evidence of usefulness, not proof of causality.

### Aporte_Marginal

For every candidate with two or more factors, the report locates the matching parent candidate with one factor removed and the same cohort size.

Example:

```text
MOMENTUM + VOLUME + STRUCTURE
minus STRUCTURE
=
MOMENTUM + VOLUME
```

The difference is recorded for:

- evolution score;
- validation Rare Event F1;
- balanced accuracy;
- Macro F1;
- Brier Skill;
- calibration error;
- generalization gap.

A positive evolution-score difference does not automatically imply that every underlying metric improved.

### Interacciones

Aggregates marginal additions by context.

Example:

```text
factor_added = VOLUME
context_factor = MOMENTUM
```

This measures whether adding Volume tends to help candidates that already contain Momentum.

### Sinergias_Pares

Compares each two-factor candidate with the average of its two corresponding single-factor candidates at the same cohort size.

A pair is marked `beats_both_singles` only when its evolution score exceeds both individual factors.

### Saturacion

Summarizes candidates by number of factors:

- average, median and best evolution score;
- score dispersion;
- number of Top-N candidates;
- validation Rare Event F1;
- balanced accuracy;
- generalization gap;
- complexity penalty;
- incremental best and average score relative to the previous factor count.

This sheet is the primary evidence for deciding whether four factors add useful information beyond three.

### Top_Candidatos

Preserves the Top N candidates used by the report for direct audit.

## Interpretation principles

- Prefer marginal contribution over raw presence when judging whether a factor adds new information.
- Prefer repeated positive contributions across cohort sizes over a single exceptional candidate.
- Treat negative marginal calibration error carefully: lower calibration error is beneficial.
- A factor may be useful alone but redundant when another factor is present.
- A pair may be synergistic in one cohort size and harmful in another.
- Saturation is provisional until confirmed on future, untouched dates.

## Scientific boundary

This report explains the historical v0.9 search. It does not validate the selected factors on a new market period. Its findings must be treated as hypotheses for subsequent walk-forward confirmation.
