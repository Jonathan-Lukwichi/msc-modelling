# pipeline_configs/

Hydra config root for the Ch6 refactor (planned, Prompt 3).

Expected structure once Prompt 3 lands:

```
pipeline_configs/
  config.yaml
  paths.yaml
  splits.yaml
  cv/      expanding.yaml  sliding_365.yaml  sliding_450.yaml  sliding_450_rulsif.yaml
  model/   arima.yaml  sarimax.yaml  negbin.yaml  xgboost.yaml  ann.yaml  lstm.yaml
           hybrid_*.yaml  deepar.yaml  direct_xgb.yaml
  hpo/     grid.yaml  random.yaml  optuna.yaml
  stage/   hpo.yaml  val.yaml  test.yaml
```

For now this directory is a placeholder created by Prompt 0 scaffolding.
