"""Hybrid mixed-precision assignment for layer-wise LLM quantization.

This is the core contribution of the project. The three per-method analyzers
(``smoothquant/smoothquant/layer_analysis.py``,
``QuaRot/quant/layer_analysis_quarot.py`` and
``SpinQuant/eval_utils/layer_analysis_spinquant.py``) each emit a per-submodule
cosine-similarity / MSE diagnostic describing how faithfully that method
reconstructs the original FP16 weights. Those diagnostics are merged into a
single table (``Outputs/CSV-FILES/merged_layerwise_analysis.csv``).

``hybrid.py`` consumes that merged table and answers the question the project is
actually about: *not every submodule tolerates aggressive quantization equally,
so which method and which precision should each submodule get?*

For every submodule it:
  1. picks the method with the highest reconstruction fidelity (cosine
     similarity, ties broken by lower MSE), and
  2. assigns a precision tier -- submodules that quantize cleanly under their
     best method are sent to INT4, the more sensitive ones are kept at INT8.

The output is a mixed-precision assignment plan plus a summary of the resulting
method mix, precision mix and the average bit-width / compression vs. FP16.

Because it operates purely on the produced CSV diagnostics, it runs on a laptop
with no GPU and no model download -- the per-method analysis (which does need a
GPU) only has to be run once to produce the diagnostics.

Example
-------
    python hybrid.py \
        --input Outputs/CSV-FILES/merged_layerwise_analysis.csv \
        --output Outputs/hybrid_assignment.csv \
        --int4-threshold 0.95
"""

import argparse
import sys

import pandas as pd

# Methods recognised in the merged diagnostics table. The merged CSV uses the
# convention "<metric>_<method>", e.g. "cosine_similarity_quarot".
METHODS = ("smoothquant", "quarot", "spinquant")

# Bit-widths used for the size / compression estimate.
PRECISION_BITS = {"INT4": 4, "INT8": 8}
FP16_BITS = 16


def _column(metric, method):
    return f"{metric}_{method}"


def _available_methods(df):
    """Return the methods that actually have a cosine-similarity column."""
    return [m for m in METHODS if _column("cosine_similarity", m) in df.columns]


def assign_row(row, methods, int4_threshold):
    """Pick the best method and a precision tier for a single submodule.

    Best method = highest cosine similarity; ties broken by lower MSE. A
    submodule whose best-method cosine clears ``int4_threshold`` is considered
    robust enough for INT4, otherwise it is kept at INT8.
    """
    best_method = None
    best_cos = -2.0
    best_mse = float("inf")
    for method in methods:
        cos = row.get(_column("cosine_similarity", method))
        mse = row.get(_column("mse", method))
        if cos is None or pd.isna(cos):
            continue
        cos = float(cos)
        mse = float(mse) if mse is not None and not pd.isna(mse) else float("inf")
        if cos > best_cos or (cos == best_cos and mse < best_mse):
            best_method, best_cos, best_mse = method, cos, mse

    if best_method is None:
        return None

    precision = "INT4" if best_cos >= int4_threshold else "INT8"
    return {
        "layer_idx": row.get("layer_idx"),
        "layer_name": row.get("layer_name"),
        "submodule": row.get("submodule"),
        "chosen_method": best_method,
        "chosen_cosine": round(best_cos, 6),
        "chosen_mse": best_mse if best_mse != float("inf") else None,
        "precision": precision,
    }


def build_assignment(df, int4_threshold):
    """Build the full mixed-precision assignment plan from the merged table."""
    methods = _available_methods(df)
    if not methods:
        raise ValueError(
            "No cosine_similarity_<method> columns found in the input CSV. "
            f"Expected one of: {', '.join(_column('cosine_similarity', m) for m in METHODS)}"
        )
    rows = [assign_row(row, methods, int4_threshold) for _, row in df.iterrows()]
    return pd.DataFrame([r for r in rows if r is not None])


def summarize(plan):
    """Print a human-readable summary of the assignment plan."""
    n = len(plan)
    if n == 0:
        print("No submodules were assigned -- is the input table empty?")
        return

    avg_bits = plan["precision"].map(PRECISION_BITS).mean()
    compression = FP16_BITS / avg_bits

    print("\nHybrid mixed-precision assignment")
    print("=" * 34)
    print(f"Submodules assigned : {n}")
    print(f"Average weight bits : {avg_bits:.2f} (vs {FP16_BITS} for FP16)")
    print(f"Weight compression  : {compression:.2f}x vs FP16")

    print("\nMethod mix:")
    for method, count in plan["chosen_method"].value_counts().items():
        print(f"  {method:<12} {count:>4}  ({100 * count / n:5.1f}%)")

    print("\nPrecision mix:")
    for precision, count in plan["precision"].value_counts().items():
        print(f"  {precision:<12} {count:>4}  ({100 * count / n:5.1f}%)")

    sensitive = plan[plan["precision"] == "INT8"]
    if not sensitive.empty:
        print("\nMost sensitive submodules (kept at INT8, lowest cosine first):")
        cols = ["layer_idx", "submodule", "chosen_method", "chosen_cosine"]
        print(sensitive.sort_values("chosen_cosine").head(10)[cols].to_string(index=False))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default="Outputs/CSV-FILES/merged_layerwise_analysis.csv",
        help="Merged per-method diagnostics CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="Outputs/hybrid_assignment.csv",
        help="Where to write the assignment plan (default: %(default)s).",
    )
    parser.add_argument(
        "--int4-threshold",
        type=float,
        default=0.95,
        help="Min best-method cosine similarity for a submodule to be sent to "
        "INT4; below this it is kept at INT8 (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Input not found: {args.input}", file=sys.stderr)
        print(
            "Run the per-method analyzers first to produce the diagnostics, "
            "then merge them (see README).",
            file=sys.stderr,
        )
        return 1

    plan = build_assignment(df, args.int4_threshold)
    plan.to_csv(args.output, index=False)
    print(f"Wrote assignment plan for {len(plan)} submodules to {args.output}")
    summarize(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
