"""
Generate the Mumbai Uber-surge teaching datasets.

Outputs
-------
data/uber_surge_mumbai.csv       main dataset, n=200, p=8 usable features
data/new_city_features.csv       p >> n dataset for the Lasso task (n=60, p=120)
site/assets/js/data.js           same data, embedded for the browser lab

The generating process is fixed by seed so the CSV, the notebooks and the
website all agree row-for-row.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

ZONES = ["BKC", "Andheri East", "Lower Parel", "Powai", "Colaba",
         "Bandra West", "Dadar", "Navi Mumbai", "Thane", "CSMIA T2"]

# Ground-truth coefficients of the data generating process.  Students never see
# these, but every task is written so that a correct solution recovers them.
TRUE = {
    "is_peak": 40.0,
    "is_rain": 30.0,
    "is_event_nearby": 20.0,
    "is_airport_pickup": 25.0,
    "traffic_penalty_per_kmph": 2.0,   # applied to (60 - traffic_speed)
    "driver_scarcity": 50.0,           # applied to 1/(drivers+1)
    "is_weekend": 0.0,                 # deliberately null -> Lasso kills it first
    "noise_sigma": 10.0,
}


def make_main(n=200, seed=SEED):
    rng = np.random.default_rng(seed)

    hour = rng.integers(0, 24, n)
    is_peak = (((hour >= 8) & (hour <= 10)) | ((hour >= 18) & (hour <= 21))).astype(int)
    is_weekend = rng.binomial(1, 0.3, n)

    # Rain is MORE likely during peak hours -> is_peak and is_rain are correlated.
    # This collinearity is the whole point of the Ridge task.
    p_rain = np.clip(0.22 + 0.30 * is_peak, 0, 1)
    is_rain = rng.binomial(1, p_rain, n)

    is_event = rng.binomial(1, 0.12 + 0.10 * is_weekend, n)
    is_airport = rng.binomial(1, 0.10, n)

    # Traffic slows in peak and rain -> correlated with both binary flags.
    traffic = rng.normal(46 - 14 * is_peak - 9 * is_rain, 4.5)
    traffic = np.clip(traffic, 5, 60).round(1)

    drivers = rng.poisson(np.clip(20 - 8 * is_peak - 5 * is_rain + 5 * is_weekend, 1, None))
    drivers = np.clip(drivers, 1, 40)

    requests = rng.poisson(np.clip(15 + 10 * is_peak + 6 * is_event + 3 * is_rain, 1, None))
    requests = np.clip(requests, 1, 80)

    historical = np.clip(requests + rng.normal(0, 3, n), 1, None).round(1)

    distance = rng.uniform(2, 20, n).round(2)
    time_min = np.clip(distance * 2.0 + 12 * is_peak + rng.normal(0, 3, n), 3, None).round(1)

    surge = (TRUE["is_peak"] * is_peak
             + TRUE["is_rain"] * is_rain
             + TRUE["is_event_nearby"] * is_event
             + TRUE["is_airport_pickup"] * is_airport
             + TRUE["driver_scarcity"] / (drivers + 1)
             + TRUE["traffic_penalty_per_kmph"] * (60 - traffic)
             + rng.normal(0, TRUE["noise_sigma"], n))
    surge = np.clip(surge, 0, 150).round(1)

    base = (40 + 12 * distance + 2 * time_min).round(1)

    df = pd.DataFrame({
        "trip_id": np.arange(1, n + 1),
        "zone": rng.choice(ZONES, n),
        "hour": hour,
        "is_peak": is_peak,
        "is_weekend": is_weekend,
        "is_rain": is_rain,
        "is_event_nearby": is_event,
        "is_airport_pickup": is_airport,
        "traffic_speed_kmph": traffic,
        "drivers_available_500m": drivers,
        "open_requests_500m": requests,
        "historical_demand": historical,
        "distance_km": distance,
        "time_min": time_min,
        "base_fare_inr": base,
        "surge_additive_inr": surge,
    })
    df["final_fare_inr"] = (df.base_fare_inr + df.surge_additive_inr).round(1)
    df["demand_supply_ratio"] = (df.open_requests_500m / df.drivers_available_500m).round(3)
    # Derived feature used in the Elastic Net "correlated group" task.
    df["is_bad_weather"] = ((df.is_rain == 1) & (df.traffic_speed_kmph < 25)).astype(int)
    return df


def make_new_city(n=60, p=120, k=6, seed=SEED + 1):
    """p >> n dataset: 120 candidate features from the city-ops data warehouse,
    only 6 of which actually drive surge."""
    rng = np.random.default_rng(seed)
    names = (["is_peak", "is_rain", "is_airport_pickup", "is_event_nearby",
              "traffic_index", "driver_scarcity_index"]
             + [f"feat_{i:03d}" for i in range(1, p - k + 1)])
    X = rng.normal(0, 1, (n, p))
    # Give the signal features a correlated partner each (decoys for Lasso).
    for j in range(k):
        X[:, k + j] = 0.85 * X[:, j] + rng.normal(0, 0.35, n)
    beta = np.zeros(p)
    beta[:k] = [38, 27, 24, 19, -16, 21]
    y = X @ beta + rng.normal(0, 8, n)
    df = pd.DataFrame(X.round(4), columns=names)
    df.insert(0, "trip_id", np.arange(1, n + 1))
    df["surge_additive_inr"] = y.round(2)
    truth = pd.DataFrame({"feature": names, "true_coefficient": beta.round(2)})
    return df, truth


def main():
    df = make_main()
    os.makedirs(f"{ROOT}/data", exist_ok=True)
    df.to_csv(f"{ROOT}/data/uber_surge_mumbai.csv", index=False)

    nc, truth = make_new_city()
    nc.to_csv(f"{ROOT}/data/new_city_features.csv", index=False)
    truth.to_csv(f"{ROOT}/data/new_city_truth.csv", index=False)

    # Browser copy: compact JSON-in-JS so the site works from file:// with no server.
    cols = [c for c in df.columns]
    payload = {
        "columns": cols,
        "rows": [[(v.item() if hasattr(v, "item") else v) for v in r] for r in df.to_numpy()],
        "truth": TRUE,
    }
    # The p >> n dataset powers the "new city launch" panel in the lab.
    nc_cols = [c for c in nc.columns if c not in ("trip_id",)]
    newcity = {
        "columns": nc_cols,
        "rows": [[round(float(v), 4) for v in r] for r in nc[nc_cols].to_numpy()],
        "truth": {r.feature: float(r.true_coefficient)
                  for r in truth.itertuples() if r.true_coefficient != 0},
    }
    js = ("// AUTO-GENERATED by scripts/make_data.py - do not edit by hand.\n"
          "window.SURGE_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
          "window.NEWCITY_DATA = " + json.dumps(newcity, separators=(",", ":")) + ";\n")
    os.makedirs(f"{ROOT}/site/assets/js", exist_ok=True)
    with open(f"{ROOT}/site/assets/js/data.js", "w") as f:
        f.write(js)

    print("main dataset:", df.shape)
    print(df.head(8).to_string(index=False))
    print("\ncorrelations (the collinearity students must handle):")
    num = ["is_peak", "is_rain", "traffic_speed_kmph", "drivers_available_500m",
           "is_bad_weather", "surge_additive_inr"]
    print(df[num].corr().round(3).to_string())
    print("\nnew-city dataset:", nc.shape)
    print("\nwrote data/*.csv and site/assets/js/data.js")


if __name__ == "__main__":
    main()
