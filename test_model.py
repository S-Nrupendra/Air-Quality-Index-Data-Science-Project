from xgboost import XGBRegressor

model = XGBRegressor()
model.load_model("best_model_Without_Xylene.json")
print(model.get_booster().feature_names)
