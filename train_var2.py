import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline


# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv("BT2024177_train_var2.csv")

X = train_df.drop(columns=["y"]).values
y = train_df["y"].values


# ============================================================
# 2. CROSS-VALIDATION SETUP
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 3. HYPERPARAMETER RANGES
# ============================================================

degrees = range(2, 15)

ridge_alphas = np.logspace(-3, 1.5, 19)

lasso_alphas = np.logspace(-4, -1, 7)


# ============================================================
# 4. CREATE MODEL
# ============================================================

def make_model(model_type, degree, alpha):

    if model_type == "ridge":
        regressor = Ridge(alpha=alpha)

    else:
        regressor = Lasso(
            alpha=alpha,
            max_iter=50000,
            tol=1e-3
        )

    return Pipeline([
        ("scaler1", StandardScaler()),
        ("poly", PolynomialFeatures(
            degree=degree,
            include_bias=False
        )),
        ("scaler2", StandardScaler()),
        ("reg", regressor)
    ])


# ============================================================
# 5. SEARCH FOR BEST ALPHA FOR EACH DEGREE
# ============================================================

def search_model(model_type, alphas):

    results = []

    for degree in degrees:

        best_mse = float("inf")
        best_alpha = None
        best_r2 = None

        for alpha in alphas:

            model = make_model(
                model_type,
                degree,
                alpha
            )

            cv = cross_validate(
                model,
                X,
                y,
                cv=kf,
                scoring={
                    "mse": "neg_mean_squared_error",
                    "r2": "r2"
                },
                n_jobs=-1
            )

            mse = -cv["test_mse"].mean()
            r2 = cv["test_r2"].mean()

            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha
                best_r2 = r2

        results.append([
            model_type,
            degree,
            best_alpha,
            best_mse,
            best_r2
        ])

        print(
            f"{model_type.capitalize():6s} | "
            f"Degree {degree:2d} | "
            f"Alpha {best_alpha:.5f} | "
            f"CV MSE {best_mse:.6f} | "
            f"CV R2 {best_r2:.4f}"
        )

    return results


# ============================================================
# 6. RUN RIDGE AND LASSO SEARCH
# ============================================================

results = []

results += search_model(
    "ridge",
    ridge_alphas
)

results += search_model(
    "lasso",
    lasso_alphas
)


# ============================================================
# 7. STORE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Degree",
        "Alpha",
        "CV_MSE",
        "CV_R2"
    ]
)


print("\n================ ALL RESULTS ================\n")
print(results_df.to_string(index=False))


# ============================================================
# 8. BEST RIDGE AND LASSO
# ============================================================

for model_name in ["ridge", "lasso"]:

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    best = model_results.loc[
        model_results["CV_MSE"].idxmin()
    ]

    print(f"\nBest {model_name.capitalize()}:")
    print(f"Degree : {int(best['Degree'])}")
    print(f"Alpha  : {best['Alpha']}")
    print(f"CV MSE : {best['CV_MSE']}")
    print(f"CV R2  : {best['CV_R2']}")


# ============================================================
# 9. OVERALL BEST MODEL
# ============================================================

best_overall = results_df.loc[
    results_df["CV_MSE"].idxmin()
]

print("\n================ OVERALL BEST ================\n")

print(f"Model  : {best_overall['Model']}")
print(f"Degree : {int(best_overall['Degree'])}")
print(f"Alpha  : {best_overall['Alpha']}")
print(f"CV MSE : {best_overall['CV_MSE']}")
print(f"CV R2  : {best_overall['CV_R2']}")


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df.to_csv(
    "partb_ridge_lasso_all_degrees.csv",
    index=False
)


# ============================================================
# 11. PLOT CV MSE VS DEGREE
# ============================================================

plt.figure(figsize=(7, 4))

for model_name in ["ridge", "lasso"]:

    model_results = results_df[
        results_df["Model"] == model_name
    ]

    plt.plot(
        model_results["Degree"],
        model_results["CV_MSE"],
        marker="o",
        label=model_name
    )

plt.ylim(0.2, 1.0)
plt.xlabel("Polynomial Degree")
plt.ylabel("5-Fold CV MSE")
plt.title("Ridge vs Lasso")
plt.legend()
plt.grid(True)

plt.savefig(
    "partb_ridge_vs_lasso.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()