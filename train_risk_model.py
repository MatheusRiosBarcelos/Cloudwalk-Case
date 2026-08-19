import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

FEATURE_STORE_PATH = "fs_transactional_data.csv"
MODEL_OUT_PATH = "risk_model.joblib"
SCORED_TEST_OUT_PATH = "risk_scores_test.csv"

BASE_FEATURES = ["transaction_amount", "hour"]
VELOCITY_FEATURES = [
    "txn_count_1h_user_id", "txn_count_24h_user_id", "txn_count_7D_user_id",
    "txn_count_1h_card_number", "txn_count_24h_card_number", "txn_count_7D_card_number",
    "txn_count_1h_device_id", "txn_count_24h_device_id", "txn_count_7D_device_id",
]
DISTINCT_ENTITY_FEATURES = ["distinct_devices_24h_user", "distinct_merchants_24h_card"]
NEW_ENTITY_FEATURES = ["is_first_txn_user", "is_first_txn_card", "is_first_txn_device"]
AMOUNT_RATIO_FEATURES = ["user_id_amount_ratio", "card_number_amount_ratio", "device_id_amount_ratio"]
RISK_RATE_FEATURES = ["merchant_cbk_rate_oot", "bin_cbk_rate_oot"]

FEATURES = (
    BASE_FEATURES + VELOCITY_FEATURES + DISTINCT_ENTITY_FEATURES
    + NEW_ENTITY_FEATURES + AMOUNT_RATIO_FEATURES + RISK_RATE_FEATURES
)


def out_of_time_rate(df: pd.DataFrame, group_col: str, target_col: str = "has_cbk") -> pd.Series:
    """Expanding chargeback rate for `group_col`, using only transactions strictly
    before the current one. The feature store's own `merchant_CBK_rate` is a
    full-period group mean (includes future transactions and the transaction
    itself) so it leaks the label - this recomputes it as of each transaction's
    own timestamp instead. Cold starts (first-ever transaction for a group)
    fall back to the running global rate at that point in time.
    """
    group_rate = (
        df.groupby(group_col)[target_col]
        .expanding()
        .mean()
        .shift()
        .reset_index(level=0, drop=True)
    )
    global_rate = df[target_col].expanding().mean().shift()
    return group_rate.fillna(global_rate).fillna(df[target_col].mean())


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_STORE_PATH, sep=";")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df = df.sort_values("transaction_date").reset_index(drop=True)
    df["has_cbk"] = df["has_cbk"].astype(int)
    df["bin"] = df["card_number"].astype(str).str[:6]

    df["merchant_cbk_rate_oot"] = out_of_time_rate(df, "merchant_id")
    df["bin_cbk_rate_oot"] = out_of_time_rate(df, "bin")

    for col in NEW_ENTITY_FEATURES:
        df[col] = df[col].astype(bool).astype(int)

    return df


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Split chronologically (train = past, test = future) rather than randomly.
    A random split would leak information: the same user/card/device often
    appears in both halves, so a model could partly memorize entities instead
    of learning generalizable risk patterns - chronological split mirrors how
    the model would actually be used in production.
    """
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_date = df["transaction_date"].iloc[cutoff_idx]
    train = df[df["transaction_date"] < cutoff_date]
    test = df[df["transaction_date"] >= cutoff_date]
    return train, test


def pick_tier_thresholds(y_true, scores, decline_recall_target=0.6, approve_fpr_target=0.05):
    """Data-driven thresholds for the three-tier policy:
    - high_threshold: score above which we still catch `decline_recall_target`
      of the test set's chargebacks (manual review / decline tier).
    - low_threshold: score below which no more than `approve_fpr_target` of
      genuine transactions are being flagged (auto-approve tier).
    Everything in between gets step-up authentication (3DS/OTP) - proportional
    friction instead of one hard cutoff.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    scores_sorted = scores[order]
    total_pos = y_sorted.sum()

    cum_pos = np.cumsum(y_sorted)
    recall = cum_pos / total_pos if total_pos else np.zeros_like(cum_pos, dtype=float)
    hi_idx = min(int(np.searchsorted(recall, decline_recall_target)), len(scores_sorted) - 1)
    high_threshold = float(scores_sorted[hi_idx])

    neg_scores = np.sort(scores[y_true == 0])
    lo_idx = min(int(len(neg_scores) * (1 - approve_fpr_target)), len(neg_scores) - 1)
    low_threshold = float(neg_scores[lo_idx]) if len(neg_scores) else 0.0

    if low_threshold > high_threshold:
        low_threshold = high_threshold

    return low_threshold, high_threshold


def tier_for_score(score: float, low: float, high: float) -> str:
    if score >= high:
        return "Manual review / decline"
    if score >= low:
        return "Step-up authentication"
    return "Auto-approve"


def main():
    df = load_features()
    train, test = time_based_split(df, test_frac=0.2)

    X_train, y_train = train[FEATURES], train["has_cbk"]
    X_test, y_test = test[FEATURES], test["has_cbk"]

    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    test_scores = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    low_thr, high_thr = pick_tier_thresholds(y_test.values, test_scores)

    perm = permutation_importance(
        model, X_test, y_test, n_repeats=15, random_state=42, scoring="average_precision"
    )
    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 60)
    print(f"Train rows: {len(train)}  ({y_train.mean():.1%} chargeback)")
    print(f"Test rows:  {len(test)}  ({y_test.mean():.1%} chargeback)")
    print(f"Test ROC-AUC: {roc_auc:.3f}")
    print(f"Test PR-AUC:  {pr_auc:.3f}  (baseline = {y_test.mean():.3f})")
    print(f"Tier thresholds -> auto-approve < {low_thr:.3f} <= step-up < {high_thr:.3f} <= decline")
    print("\nTop features (permutation importance, scored on PR-AUC):")
    print(importance.head(10).to_string(index=False))

    tiers = pd.Series(test_scores, index=test.index).apply(lambda s: tier_for_score(s, low_thr, high_thr))
    tier_summary = (
        pd.DataFrame({"tier": tiers.values, "has_cbk": y_test.values})
        .groupby("tier")["has_cbk"]
        .agg(transactions="count", chargeback_rate="mean")
        .reindex(["Auto-approve", "Step-up authentication", "Manual review / decline"])
    )
    print("\nThree-tier decision breakdown (test set):")
    print(tier_summary.to_string())

    scored_test = test[
        ["transaction_id", "transaction_date", "merchant_id", "user_id", "card_number", "device_id",
         "transaction_amount", "has_cbk"]
    ].copy()
    scored_test["risk_score"] = test_scores
    scored_test["tier"] = tiers.values
    scored_test.to_csv(SCORED_TEST_OUT_PATH, index=False)
    print(f"Saved scored test set to {SCORED_TEST_OUT_PATH}")

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "low_threshold": low_thr,
            "high_threshold": high_thr,
            "test_roc_auc": float(roc_auc),
            "test_pr_auc": float(pr_auc),
            "feature_importance": importance,
            "test_size": len(test),
            "test_cbk_rate": float(y_test.mean()),
            "trained_at": pd.Timestamp.now().isoformat(),
        },
        MODEL_OUT_PATH,
    )
    print(f"\nSaved model + thresholds to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
