"""
Predictive Modelling – Lead Time Prediction
Nassau Candy Distributor - Factory Reallocation System
"""
import pickle, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_prep import (
    load_and_prepare, build_feature_matrix,
    FACTORIES, PRODUCT_FACTORY, REGION_COORDS, haversine_km, SHIP_MODE_BASE
)

MODEL_PATH   = "models/best_model.pkl"
ENCODERS_PATH= "models/encoders.pkl"


def train_and_evaluate(path: str):
    df = load_and_prepare(path)
    X, y, encoders = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=300, max_depth=12,
                                                    random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                        learning_rate=0.05, random_state=42),
    }

    results, trained = {}, {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae   = float(mean_absolute_error(y_test, preds))
        r2    = float(r2_score(y_test, preds))
        results[name] = {"RMSE": round(rmse,3), "MAE": round(mae,3), "R2": round(r2,4)}
        trained[name] = mdl
        print(f"{name:25s}  RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}")

    best_name  = min(results, key=lambda k: results[k]["RMSE"])
    best_model = trained[best_name]
    print(f"\n✓ Best model: {best_name}")

    os.makedirs("models", exist_ok=True)
    pickle.dump(best_model, open(MODEL_PATH, "wb"))
    pickle.dump({"encoders": encoders, "best_name": best_name,
                 "all_results": results, "all_models": trained},
                open(ENCODERS_PATH, "wb"))
    return best_model, encoders, results, best_name, df


def load_model():
    model    = pickle.load(open(MODEL_PATH, "rb"))
    enc_data = pickle.load(open(ENCODERS_PATH, "rb"))
    return (model, enc_data["encoders"], enc_data["best_name"],
            enc_data["all_results"], enc_data.get("all_models", {}))


def predict_lead_time(model, encoders, product, region, factory, ship_mode,
                      units=3, cost=3.5, order_month=6, order_dow=1):
    f = FACTORIES[factory];  r = REGION_COORDS[region]
    dist = haversine_km(f["lat"], f["lon"], r["lat"], r["lon"])
    row  = np.array([[
        encoders["product"].transform([product])[0],
        encoders["region"].transform([region])[0],
        encoders["factory"].transform([factory])[0],
        encoders["mode"].transform([ship_mode])[0],
        dist, units, cost, order_month, order_dow
    ]])
    return float(model.predict(encoders["scaler"].transform(row))[0])


def simulate_factory_reassignment(model, encoders, df, product, region, ship_mode):
    current_factory = PRODUCT_FACTORY[product]
    rows = []
    for factory in FACTORIES:
        lt   = predict_lead_time(model, encoders, product, region, factory, ship_mode)
        mask = (df["Product Name"] == product) & (df["Region"] == region)
        sub  = df[mask]
        avg_gp  = float(sub["Gross Profit"].mean() if len(sub) else df["Gross Profit"].mean())
        avg_mg  = float(sub["Profit_Margin"].mean() if len(sub) else df["Profit_Margin"].mean())
        f_info  = FACTORIES[factory]; r_info = REGION_COORDS[region]
        dist    = haversine_km(f_info["lat"], f_info["lon"], r_info["lat"], r_info["lon"])
        rows.append({
            "Factory": factory, "Is Current": factory == current_factory,
            "Predicted LT": round(lt, 2), "Distance km": round(dist, 0),
            "Avg Gross Profit": round(avg_gp, 2), "Avg Margin %": round(avg_mg, 1),
        })

    sim = pd.DataFrame(rows).sort_values("Predicted LT")
    cur_lt = sim[sim["Is Current"]]["Predicted LT"].values[0]
    sim["LT Improvement"]  = (cur_lt - sim["Predicted LT"]).round(2)
    sim["Improvement %"]   = ((cur_lt - sim["Predicted LT"]) / max(cur_lt, 0.01) * 100).round(1)
    sim["Confidence"]      = (90 - sim["Distance km"] / 60).clip(50, 95).round(1)
    return sim


def generate_top_recommendations(model, encoders, df, top_n=20):
    rows = []
    for product in PRODUCT_FACTORY:
        for region in REGION_COORDS:
            for ship_mode in ["Standard Class", "First Class", "Second Class", "Same Day"]:
                sim     = simulate_factory_reassignment(model, encoders, df, product, region, ship_mode)
                current = sim[sim["Is Current"]].iloc[0]
                best    = sim.iloc[0]
                if not best["Is Current"] and best["LT Improvement"] > 0:
                    rows.append({
                        "Product":         product,
                        "Region":          region,
                        "Ship Mode":       ship_mode,
                        "Current Factory": PRODUCT_FACTORY[product],
                        "Recommended":     best["Factory"],
                        "Current LT":      current["Predicted LT"],
                        "New LT":          best["Predicted LT"],
                        "LT Reduction":    best["LT Improvement"],
                        "LT Reduction %":  best["Improvement %"],
                        "Dist km":         best["Distance km"],
                        "Avg Profit":      best["Avg Gross Profit"],
                        "Confidence":      best["Confidence"],
                    })
    return pd.DataFrame(rows).sort_values("LT Reduction %", ascending=False).head(top_n)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Training models…\n")
    model, encoders, results, best_name, df = train_and_evaluate("data/Nassau_Candy_Distributor.csv")
    print("\nAll results:", results)
