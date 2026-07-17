# v0.9 — Evolutionary Universe Laboratory

## Research question

Can the platform discover a conditioned universe that generalizes better than manually selected universes without allowing the search space to become unbounded?

## Scope

The first laboratory is deliberately constrained:

- six causal factors;
- combinations containing one to three factors;
- Top 5, Top 10 and Top 20 cohorts;
- the same historical predictions for every candidate;
- chronological discovery and validation periods;
- complexity and generalization penalties;
- a diverse Hall of Fame.

With six factors and combinations of at most three factors, the default search contains 41 factor combinations. Across three cohort sizes this produces 123 candidates. This is large enough to discover interactions while remaining auditable and computationally manageable.

## Available factors

### MOMENTUM

Cross-sectional strength based on returns over one, three and five sessions.

### VOLUME

Relative volume combined with the closing position inside the daily range.

### SMART_MONEY

Current Smart Money percentage and its five-session slope.

### MODEL_CONFIDENCE

The highest predicted probability among GAP_UP, GAP_DOWN and SIN_GAP.

### LOW_VOLATILITY

Preference for lower ATR inside the daily cross-section.

### STRUCTURE

Closing price relative to VWAP and the closing position inside the daily range.

All factor scores use information available at the close of `origin_date`.

## Candidate construction

A candidate is defined by:

1. a factor signature, such as `MOMENTUM+VOLUME`;
2. an equal-weight average of its factor scores;
3. a daily cohort size, such as Top 10.

Equal weights are intentional in v0.9. Optimizing continuous weights would multiply the search space and make interpretation more difficult before the factor interactions have been understood.

## Chronological validation

The available origin dates are divided chronologically:

- discovery period: first 70% by default;
- validation period: final 30% by default.

No random shuffling is used. Candidate quality is ranked primarily with validation metrics.

## Evolution score

The validation quality score ranks candidates using:

- Rare Event F1;
- Balanced Accuracy;
- Macro F1;
- Brier Skill;
- calibration error.

Two penalties are then applied:

- complexity penalty: additional factors must justify their existence;
- generalization penalty: candidates that change sharply between discovery and validation are downgraded.

The resulting `evolution_score` is a comparative research score. It is not a probability, return forecast or trading recommendation.

## Diverse Hall of Fame

The highest score alone can produce many near-duplicate candidates. The Hall of Fame therefore preserves a limited set of strong but structurally different factor signatures. Its purpose is to identify several promising research families rather than declaring a single permanent winner.

## Guardrails

The v0.9 laboratory does not:

- modify the predictor;
- use future outcomes to construct daily cohorts;
- optimize arbitrary continuous weights;
- test thousands of unconstrained generations;
- treat the best in-sample candidate as confirmed knowledge;
- incorporate Smart Money inertia summaries calculated with future dates.

## Required confirmation

A candidate is only a discovery hypothesis until it succeeds in a later untouched period or a subsequent market regime. The v0.9 workbook is therefore a discovery instrument, not final proof.

## Outputs

- `Leaderboard`: all candidates ordered by evolutionary score;
- `Hall_of_Fame`: strong and diverse candidate families;
- `Metricas_Candidatos`: discovery and validation metrics;
- `Definiciones`: exact candidate signatures and sizes;
- `Selecciones`: daily ticker membership and scores;
- `Diagnostico_Diario`: cohort density and rare-event counts by date.

## Next scientific questions

The results should help answer:

1. Which factors repeatedly survive validation?
2. Does factor interaction add value beyond the best individual factor?
3. Is Top 5, Top 10 or Top 20 more stable outside discovery?
4. Do simpler universes generalize better?
5. Are the winning factors stable across later regimes?
6. Which causal daily Smart Money regime variables should enter the next generation?
