import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def load_data():
    train_df = pd.read_csv("BT2024177_train_var2.csv")
    test_df = pd.read_csv("BT2024177_test_var2.csv")

    feature_cols = [c for c in train_df.columns if c != "y"]

    X = train_df[feature_cols].values
    y = train_df["y"].values
    X_test = test_df[feature_cols].values

    return X, y, X_test

def make_model(alpha, degree=10):
    return Pipeline([
        ("scaler1", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler2", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])


def find_best_alpha(X, y, alphas, kf, degree=10):
    best_alpha = None
    best_mse = float("inf")
    best_r2 = None

    for alpha in alphas:
        cv = cross_validate(
            make_model(alpha, degree), X, y, cv=kf,
            scoring={"mse": "neg_mean_squared_error", "r2": "r2"}
        )
        mse = -cv["test_mse"].mean()
        r2 = cv["test_r2"].mean()
        print(f"alpha {alpha:.4f}: CV MSE = {mse:.4f}, CV R2 = {r2:.4f}")

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_r2 = r2

    return best_alpha, best_mse, best_r2


def calculate_metrics(y, predictions):
    mse = np.mean((y - predictions) ** 2)
    r2 = 1 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)
    return mse, r2


def train_final_model(X, y, alpha, degree=10):
    model = make_model(alpha, degree)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y, name):
    predictions = model.predict(X)
    mse, r2 = calculate_metrics(y, predictions)

    print(f"\n{name}:")
    print("MSE:", mse)
    print("R2:", r2)

    return predictions


def plot_residuals(X, y, alpha, degree, kf):
    oof = cross_val_predict(make_model(alpha, degree), X, y, cv=kf)

    plt.figure(figsize=(7, 4))
    plt.scatter(oof, y - oof, s=8)
    plt.axhline(0, color="k")
    plt.xlabel("predicted y")
    plt.ylabel("residual")
    plt.title("Part B: out-of-fold residuals (Ridge)")
    plt.savefig("partb_residuals.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions(predictions):
    pd.DataFrame({"y": predictions}).to_csv("BT2024177_pred_var2.csv", index=False)
    print("\nPredictions saved to BT2024177_pred_var2.csv")


def main():
    X, y, X_test = load_data()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    alphas = np.logspace(-1, 0.5, 7)
    degree = 10

    best_alpha, best_cv_mse, best_cv_r2 = find_best_alpha(X, y, alphas, kf, degree)

    print("\nBest alpha:", best_alpha)
    print("CV MSE:", best_cv_mse)
    print("CV R2:", best_cv_r2)

    final_model = train_final_model(X, y, best_alpha, degree)

    evaluate_model(final_model, X, y, "Training")

    plot_residuals(X, y, best_alpha, degree, kf)

    test_predictions = final_model.predict(X_test)
    save_predictions(test_predictions)


if __name__ == "__main__":
    main()