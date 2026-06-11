# Model Aliases — App-Facing Names

All app UI labels, URL slugs, dropdown options, badge text, and API parameter values MUST use the alias names below. **Never expose the scientific name in the UI.**

## Alias → Scientific Name Mapping

| Alias | Scientific Name | Type | Source |
|---|---|---|---|
| **Stat 1** | ARIMA(p, 1, q) | Statistical (univariate) | Thesis Chapter 5 §5.2.2 + Chapter 6 |
| **Stat 2** | SARIMAX(p, 1, q)(P, 1, Q)₇ | Statistical (with exogenous regressors) | Thesis Chapter 5 §5.2.2 + Chapter 6 |
| **ML 1** | XGBoost (Gradient Boosting Trees) | Machine Learning | Thesis Chapter 6 |
| **ML 2** | ANN (Artificial Neural Network, MLP) | Machine Learning | Thesis Chapter 6 |
| **Hybrid 1** | SARIMAX + XGBoost residual refiner | Hybrid (statistical base + ML refiner) | Thesis Chapter 6 |
| **Hybrid 2** | SARIMAX + LSTM residual refiner | Hybrid (statistical base + ML refiner) | Thesis Chapter 6 |

## Why aliases?

1. **Clinical accessibility**: doctors and hospital managers do not need to know "SARIMAX(p,1,q)(P,1,Q)₇" — they need to know which option to pick and how trustworthy it is.
2. **Forward-compatibility**: if the model behind an alias is replaced in v2 (e.g., we swap ANN for a Transformer), the alias stays the same and the UI doesn't change.
3. **Branding consistency**: the deployed product has a clean, simple identity.

## Per-Task Availability

### Task 1 (Daily Total Arrivals) — all 6 aliases available

```json
{
  "task1": ["Stat 1", "Stat 2", "ML 1", "ML 2", "Hybrid 1", "Hybrid 2"]
}
```

### Task 2 (Per Specialty) — only thesis-validated winners

```json
{
  "task2": {
    "Medicine":    ["Stat 1", "ML 1", "ML 2"],
    "Orthopaedics": ["Stat 1"],
    "Surgery":     ["ML 1", "ML 2"],
    "Gynaecology": ["Stat 1", "ML 1", "ML 2"],
    "Paediatrics": ["Stat 1", "ML 2"],
    "Maternity":   ["Stat 2"],
    "Psychiatry":  ["Stat 2"]
  }
}
```

> ⚠️ When the user selects a specialty in Task 2, the dropdown of available models **must** be filtered to only the aliases listed above. Showing an alias that isn't trained for a specialty is a deployment bug.

## Engineering reference (NOT for UI)

The internal mapping is also serialised in machine-readable form for engineering / audit:

```
_internal_only/alias_scientific_mapping.json
```

If an integrator or auditor needs to know what's behind "Hybrid 1", they look there, not in the user-facing UI.
