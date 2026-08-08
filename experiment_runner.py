from pathlib import Path
import json
import pandas as pd
from model_core import (ModelConfig, evaluate_scores, fit_anomaly_model,
                        score_with_fitted_model, run_experiment_grid, scenario_detection)

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'; OUT=BASE/'outputs'; OUT.mkdir(exist_ok=True)
sensor=pd.read_csv(DATA/'simulated_drone_sensing.csv',parse_dates=['timestamp'])
features=pd.read_csv(DATA/'image_features.csv')
cfg=ModelConfig()
fitted=fit_anomaly_model(sensor,features,cfg)
df=score_with_fitted_model(sensor,features,cfg,fitted)
metrics=evaluate_scores(df)
alpha,ablation,noise=run_experiment_grid(sensor,features,cfg)
scenario=scenario_detection(df)
df.to_csv(OUT/'baseline_scored_mission.csv',index=False)
alpha.to_csv(OUT/'alpha_sensitivity.csv',index=False)
ablation.to_csv(OUT/'modality_ablation.csv',index=False)
noise.to_csv(OUT/'noise_robustness.csv',index=False)
scenario.to_csv(OUT/'scenario_detection.csv',index=False)
with open(OUT/'baseline_metrics.json','w',encoding='utf-8') as f: json.dump(metrics,f,indent=2)
print('Baseline metrics:',metrics)
print('Saved results to',OUT)
