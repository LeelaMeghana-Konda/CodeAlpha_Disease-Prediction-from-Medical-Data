"""
main.py
-------
Entry point. Runs the full disease prediction pipeline for all three datasets
and all four classifiers. Outputs metrics tables and plots to results/.

Usage:
    python main.py
    python main.py --dataset heart_disease
    python main.py --no-cv        (skip cross-validation, faster)
    python main.py --no-plots     (skip plot generation)
"""

import argparse
import os
import sys
import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ── Resolve package root ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_loader       import DATASETS, load_all
from models.classifiers     import get_models, train_all, get_feature_importance
from models.evaluator       import (
    evaluate_all, cross_validate_all,
    results_to_dataframe, cv_to_dataframe
)
from utils.preprocessor     import split_and_scale
from utils.visualizer       import (
    plot_roc_curves, plot_confusion_matrices,
    plot_feature_importance, plot_model_comparison,
    plot_cv_comparison, plot_dashboard
)

OUTPUT_DIR = "results"


# ── Pipeline for a single dataset ─────────────────────────────────────────────

def run_dataset(dataset_key, X, y, feature_names, run_cv=True, make_plots=True):
    meta = DATASETS[dataset_key]
    dname = meta["display_name"]
    pos   = meta["positive_class"]
    neg   = meta["negative_class"]

    print(f"\n{'='*60}")
    print(f"  {dname}  ({X.shape[0]} samples · {X.shape[1]} features)")
    print(f"{'='*60}")

    # 1. Split & scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    # 2. Train
    print("\n  Training models:")
    fitted = train_all(X_train, y_train)

    # 3. Evaluate on test set
    print("\n  Evaluating on test set:")
    eval_results = evaluate_all(fitted, X_test, y_test)
    df_eval = results_to_dataframe(eval_results)
    print(df_eval.to_string(index=False))

    # 4. Cross-validation
    cv_results = None
    if run_cv:
        print("\n  Running 5-fold cross-validation:")
        raw_models = get_models()
        cv_results = cross_validate_all(raw_models, X, y, n_splits=5)
        df_cv = cv_to_dataframe(cv_results)
        print(df_cv[["Model","Accuracy (mean)","Accuracy (std)","Auc (mean)"]].to_string(index=False))

    # 5. Feature importance
    print("\n  Computing feature importance:")
    fi_data = {}
    for name, model in fitted.items():
        pairs = get_feature_importance(model, name, feature_names)
        fi_data[name] = pairs
        if pairs:
            top3 = ", ".join(f"{f}({v:.2f})" for f, v in pairs[:3])
            print(f"    {name}: {top3}")
        else:
            print(f"    {name}: N/A (RBF SVM)")

    # 6. Plots
    if make_plots:
        print("\n  Generating plots:")
        ds_dir = os.path.join(OUTPUT_DIR, dataset_key)
        plot_roc_curves(eval_results, dname, ds_dir)
        plot_confusion_matrices(eval_results, dname, (neg, pos), ds_dir)
        plot_model_comparison(eval_results, dname, ds_dir)
        plot_dashboard(eval_results, dname, ds_dir)
        if cv_results:
            plot_cv_comparison(cv_results, dname, ds_dir)
        for name, pairs in fi_data.items():
            if pairs:
                plot_feature_importance(pairs, name, dname, output_dir=ds_dir)

    # 7. Save CSV
    csv_path = os.path.join(OUTPUT_DIR, dataset_key, "metrics.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_eval.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved → {csv_path}")

    return {
        "eval":  eval_results,
        "cv":    cv_results,
        "fi":    fi_data,
        "df":    df_eval,
    }


# ── Summary across datasets ───────────────────────────────────────────────────

def print_final_summary(all_results):
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY")
    print(f"{'='*60}")
    rows = []
    for ds_key, res in all_results.items():
        dname = DATASETS[ds_key]["display_name"]
        best_row = res["df"].iloc[0]
        rows.append({
            "Dataset":    dname,
            "Best Model": best_row["Model"],
            "Accuracy":   best_row["Accuracy"],
            "AUC":        best_row["AUC"],
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Disease Prediction Pipeline")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                        help="Run only a specific dataset (default: all)")
    parser.add_argument("--no-cv",    action="store_true", help="Skip cross-validation")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n Disease Prediction — ML Classification Pipeline")
    print(" SVM · Logistic Regression · Random Forest · XGBoost\n")

    # Load data
    print("Loading datasets:")
    all_data = load_all()

    # Filter if requested
    if args.dataset:
        all_data = {args.dataset: all_data[args.dataset]}

    # Run pipeline
    all_results = {}
    for ds_key, (X, y, feature_names) in all_data.items():
        all_results[ds_key] = run_dataset(
            ds_key, X, y, feature_names,
            run_cv=not args.no_cv,
            make_plots=not args.no_plots,
        )

    # Final summary
    if len(all_results) > 1:
        print_final_summary(all_results)

    print("\n Done. All outputs in results/\n")


if __name__ == "__main__":
    main()
