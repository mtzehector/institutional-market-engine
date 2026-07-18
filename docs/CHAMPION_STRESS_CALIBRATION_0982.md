# v0.9.8.2 — Stress Calibration and Active Episode Engine

## Objective

Separate a champion's historical susceptibility to stress from the stress accumulating in its current state.

## Main corrections

1. `historical_stress_susceptibility` summarizes completed and historical episodes.
2. `current_active_stress_score` uses current lifecycle state, raw state, pending adverse transition, negative slope, recent drawdown and health deficit.
3. Deterioration velocity begins at the first weakening signal and combines damage per observation with the negative trajectory slope.
4. Relapse evaluation is censored: episodes without the complete observation horizon are excluded from the relapse denominator.
5. Recovery triggers include a coherence score based on improvement in advantage, slope and health.

## Interpretation

- High historical susceptibility with low active stress means the champion has suffered severe crises before but is not necessarily under stress now.
- Low historical susceptibility with high active stress means the current deterioration may be new and should not be hidden by a benign past.
- `stress_risk_score` combines 35% historical susceptibility, 55% active stress and 10% observable relapse risk.

## Output sheets

- `Estres_Actual`
- `Episodios_Activos`
- `Episodios_Estres`
- `Estres_Campeon`
- `Estres_Familia`
- `Coherencia_Disparadores`
- `Recaida_Censurada`
- `Resiliencia_Actual`
- `Ciclo_Actual`

The laboratory remains exploratory. Scores are not calibrated probabilities or automatic investment instructions.
