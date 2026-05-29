# Literature corpus pattern report

- Papers analysed: **119** (of 120 PDFs)
- Failed to extract text: 1 (likely scanned images)

## Most common metrics
| feature            |   n_papers |   pct_papers |
|:-------------------|-----------:|-------------:|
| metric_accuracy    |        104 |         87.4 |
| metric_recall      |         57 |         47.9 |
| metric_precision   |         54 |         45.4 |
| metric_AUC         |         42 |         35.3 |
| metric_RMSE        |         37 |         31.1 |
| metric_F1          |         36 |         30.3 |
| metric_MAPE        |         32 |         26.9 |
| metric_MAE         |         32 |         26.9 |
| metric_MSE         |         21 |         17.6 |
| metric_specificity |         20 |         16.8 |
| metric_R2          |         16 |         13.4 |
| metric_AIC_BIC     |          7 |          5.9 |
| metric_sMAPE       |          5 |          4.2 |
| metric_MASE        |          1 |          0.8 |

## Most common models compared
| feature           |   n_papers |   pct_papers |
|:------------------|-----------:|-------------:|
| model_ANN_MLP     |         90 |         75.6 |
| model_Ensemble    |         82 |         68.9 |
| model_RF          |         79 |         66.4 |
| model_ARIMA       |         70 |         58.8 |
| model_DT          |         56 |         47.1 |
| model_SVM         |         56 |         47.1 |
| model_GBM         |         53 |         44.5 |
| model_Hybrid      |         51 |         42.9 |
| model_LinReg      |         47 |         39.5 |
| model_LogReg      |         45 |         37.8 |
| model_XGBoost     |         42 |         35.3 |
| model_RNN         |         35 |         29.4 |
| model_LSTM        |         33 |         27.7 |
| model_KNN         |         32 |         26.9 |
| model_CNN         |         29 |         24.4 |
| model_SARIMA      |         26 |         21.8 |
| model_Lasso       |         24 |         20.2 |
| model_GLM         |         22 |         18.5 |
| model_NaiveBayes  |         16 |         13.4 |
| model_Prophet     |         11 |          9.2 |
| model_GRU         |         11 |          9.2 |
| model_Transformer |          8 |          6.7 |
| model_ELM         |          8 |          6.7 |
| model_Naive       |          1 |          0.8 |

## Most common presentation devices
| feature                 |   n_papers |   pct_papers |
|:------------------------|-----------:|-------------:|
| has_results_table       |         68 |         57.1 |
| has_variable_importance |         47 |         39.5 |
| has_box_plot            |         16 |         13.4 |
| has_flowchart           |         16 |         13.4 |
| has_arch_diagram        |         15 |         12.6 |
| has_confusion_matrix    |         14 |         11.8 |
| has_ACF_PACF            |         13 |         10.9 |
| has_actual_vs_pred      |         10 |          8.4 |
| has_PRISMA              |          8 |          6.7 |
| has_heatmap             |          6 |          5   |
| has_scatter_plot        |          6 |          5   |
| has_time_series_plot    |          4 |          3.4 |
| has_residual_plot       |          2 |          1.7 |

## Statistical tests used
| feature              |   n_papers |   pct_papers |
|:---------------------|-----------:|-------------:|
| test_Pearson         |          8 |          6.7 |
| test_Spearman        |          6 |          5   |
| test_ADF             |          5 |          4.2 |
| test_t_test          |          5 |          4.2 |
| test_Diebold_Mariano |          3 |          2.5 |
| test_Cointegration   |          3 |          2.5 |
| test_KS              |          2 |          1.7 |
| test_Granger         |          2 |          1.7 |
| test_Wilcoxon        |          2 |          1.7 |
| test_LjungBox        |          1 |          0.8 |

## Validation strategies
| feature               |   n_papers |   pct_papers |
|:----------------------|-----------:|-------------:|
| uses_cross_val        |         53 |         44.5 |
| uses_train_test_split |         11 |          9.2 |
| uses_train_val_test   |          5 |          4.2 |
| uses_rolling_origin   |          1 |          0.8 |

## Forecast-horizon framing
| feature         |   n_papers |   pct_papers |
|:----------------|-----------:|-------------:|
| horizon_daily   |         40 |         33.6 |
| horizon_hourly  |         20 |         16.8 |
| horizon_weekly  |         13 |         10.9 |
| horizon_monthly |         12 |         10.1 |
| horizon_multi   |          8 |          6.7 |

## Discussion-section structure
| feature              |   n_papers |   pct_papers |
|:---------------------|-----------:|-------------:|
| sec_conclusion       |         81 |         68.1 |
| sec_limitations      |         81 |         68.1 |
| sec_future_work      |         66 |         55.5 |
| sec_discussion       |         57 |         47.9 |
| sec_clinical_impl    |          6 |          5   |
| sec_comparison_prior |          0 |          0   |