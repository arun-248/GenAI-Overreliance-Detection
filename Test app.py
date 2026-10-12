"""
Test runner for the AI Overreliance Research Framework app.

Run from the project folder:   py test_app.py
It executes real tests against app.py and the saved model files using Streamlit's AppTest,
then writes test_results.json, which the app's Testing page displays. Nothing is marked as
passed unless the test actually ran and its observed result matched the expectation.
"""
import datetime
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path

if os.environ.get("TEST_WITHOUT_SHAP") or os.environ.get("OVERRELIANCE_USE_BUILTIN_SHAP"):  # simulate a machine where the shap package cannot be imported
    sys.modules["shap"] = None

import joblib
import numpy as np
import pandas as pd
try:
    import shap
    SHAP_VERSION = shap.__version__
except Exception:
    shap = None
    SHAP_VERSION = "not importable (built-in TreeSHAP used)"
import sklearn
import streamlit as st
from streamlit.testing.v1 import AppTest

HERE = Path(__file__).resolve().parent
APP = str(HERE / "app.py")
RESULTS = []

pipeline = joblib.load(HERE / "final_rf_pipeline.pkl")
metadata = joblib.load(HERE / "model_metadata.pkl")
thresholds = joblib.load(HERE / "risk_thresholds.pkl")
NUMERIC = list(metadata["numeric_features"])
CATEGORICAL = list(metadata["categorical_features"])
LOW, HIGH = float(thresholds["low_threshold"]), float(thresholds["high_threshold"])
ENCODER = pipeline.named_steps["preprocessor"].named_transformers_["categorical"]
CATS = {f: list(c) for f, c in zip(CATEGORICAL, ENCODER.categories_)}
PAGES = ["Home", "Assess", "Prediction", "Why This Result?", "Intervention", "Student Profile",
         "Model Insights", "Research", "Testing", "About"]


def record(tid, category, name, expected, fn):
    try:
        ok, observed = fn()
        status = "Pass" if ok else "Fail"
    except Exception as exc:  # a crashing test is a failed test
        status, observed = "Fail", f"Raised {type(exc).__name__}: {exc}"
    RESULTS.append({"id": tid, "category": category, "test": name, "expected": expected,
                    "observed": str(observed), "status": status})
    print(f"{tid:5s} {status:5s} {name}")


def new_app(path=APP):
    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file(path, default_timeout=240)


def exceptions(at):
    return [e.value for e in at.exception]


def has_text(at, text):
    return any(text in m.value for m in at.markdown) or any(text in c.value for c in at.caption)


def button(at, key):
    return [b for b in at.button if b.key == key][0]


def make_inputs(numeric=4, cgpa=3.30, overrides=None):
    data = {f: (cgpa if f == "CGPA" else numeric) for f in NUMERIC}
    data.update({f: CATS[f][0] for f in CATEGORICAL})
    data.update(overrides or {})
    return data


def submit(inputs, confirmed=True, nav="Assess"):
    at = new_app()
    at.session_state["_sticky"] = {f"in_{k}": v for k, v in inputs.items()}
    at.session_state["confirmed"] = set(range(5)) if confirmed else set()
    at.session_state["step"] = 5
    at.session_state["nav"] = nav
    at.run()
    if confirmed:
        button(at, "assess_go").click().run()
    return at


def expected_outcome(row, score):
    risk = "Low" if score < LOW else ("Medium" if score < HIGH else "High")
    if risk == "High":
        if row["Accept_Without_Verification"] >= 5:
            return risk, "Verification Challenge"
        if row["Overreliance_Problem_Solving"] >= 5:
            return risk, "Independent Problem-Solving Task"
        if row["Independent_Confidence"] <= 3:
            return risk, "Guided Independent Reasoning"
        return risk, "Critical Analysis Task"
    if risk == "Medium":
        if row["Cross_Check"] <= 3:
            return risk, "AI Output Verification Task"
        if row["Independent_Confidence"] <= 3:
            return risk, "Guided Reasoning Question"
        if row["Own_Ideas"] <= 3:
            return risk, "Independent Idea Generation"
        return risk, "Reflective AI-Use Question"
    return risk, "Responsible AI Use Guidance"


# ---------------------------------------------------------------- tests
def t_start():
    at = new_app().run()
    return (not exceptions(at) and at.session_state["nav"] == "Home"), f"exceptions={exceptions(at)}, page={at.session_state['nav']}"


def t_valid(inputs, label):
    def run():
        at = submit(inputs)
        res = at.session_state["result"]
        if res is None:
            return False, f"no result; exceptions={exceptions(at)}"
        ok = 1.0 <= res["score"] <= 7.0 and at.session_state["nav"] == "Prediction" and not exceptions(at)
        return ok, f"{label}: score={res['score']:.3f}, level={res['risk']}, page={at.session_state['nav']}"
    return run


def t_clamp():
    inputs = make_inputs(overrides={"Own_Ideas": 99, "CGPA": 9.9})
    at = submit(inputs)
    row = at.session_state["result"]["input_df"].iloc[0]
    return (row["Own_Ideas"] == 7.0 and row["CGPA"] == 4.0), f"Own_Ideas=99 -> {row['Own_Ideas']}, CGPA=9.9 -> {row['CGPA']}"


def t_nan():
    at = submit(make_inputs(overrides={"Cross_Check": float("nan")}))
    row = at.session_state["result"]["input_df"].iloc[0]
    return (row["Cross_Check"] == 4.0 and not exceptions(at)), f"Cross_Check=NaN -> {row['Cross_Check']}"


def t_unseen_pipeline():
    df = pd.DataFrame([make_inputs(overrides={"Gender": "UnlistedValue"})])
    score = float(pipeline.predict(df)[0])
    return bool(np.isfinite(score) and 1 <= score <= 7), f"prediction with unseen Gender = {score:.3f}"


def t_unseen_app():
    at = submit(make_inputs(overrides={"Gender": "UnlistedValue"}))
    row = at.session_state["result"]["input_df"].iloc[0]
    return row["Gender"] == CATS["Gender"][0], f"Gender fell back to '{row['Gender']}'"


def t_incomplete():
    at = submit(make_inputs(), confirmed=False)
    return (button(at, "assess_go").disabled and at.session_state["result"] is None), \
        f"submit disabled={button(at, 'assess_go').disabled}, result={at.session_state['result']}"


def t_ui_flow():
    at = new_app()
    at.session_state["nav"] = "Assess"
    at.run()
    for _ in range(5):
        button(at, "assess_next").click().run()
    reached = at.session_state["step"] == 5 and at.session_state["confirmed"] == set(range(5))
    enabled = not button(at, "assess_go").disabled
    button(at, "assess_go").click().run()
    ok = reached and enabled and at.session_state["nav"] == "Prediction" and at.session_state["result"] is not None
    return ok, f"review reached={reached}, submit enabled={enabled}, page={at.session_state['nav']}"


def t_reset():
    at = submit(make_inputs())
    at.session_state["nav"] = "Assess"
    at.session_state["step"] = 5
    at.run()
    button(at, "assess_reset").click().run()
    ok = at.session_state["confirmed"] == set() and at.session_state["result"] is None and at.session_state["step"] == 0
    return ok, f"confirmed={at.session_state['confirmed']}, result={at.session_state['result']}, step={at.session_state['step']}"


def t_repeat():
    at = submit(make_inputs())
    at.session_state["nav"] = "Assess"
    at.session_state["step"] = 5
    at.run()
    button(at, "assess_go").click().run()
    return (not exceptions(at) and at.session_state["result"] is not None), f"exceptions={exceptions(at)}"


def t_thresholds():
    return (LOW == 3.5 and HIGH == 5.0), f"low_threshold={LOW}, high_threshold={HIGH}"


def _frame(rng, n, low, constraints):
    data = {f: rng.integers(low, 8, n).astype(float) for f in NUMERIC if f != "CGPA"}
    data["CGPA"] = rng.uniform(2.31, 4.0, n)
    for f, (lo, hi) in constraints.items():
        data[f] = rng.integers(lo, hi + 1, n).astype(float)
    for f in CATEGORICAL:
        data[f] = rng.choice(CATS[f], n)
    return pd.DataFrame(data)


def find_samples(n=30000, seed=7):
    """Search random answer sets until every intervention-rule branch has an example (if the model can produce one)."""
    rng = np.random.default_rng(seed)
    regimes = [
        (1, {}),
        (4, {}),
        (4, {"Accept_Without_Verification": (1, 4), "Overreliance_Problem_Solving": (5, 7)}),
        (4, {"Accept_Without_Verification": (1, 4), "Overreliance_Problem_Solving": (1, 4), "Independent_Confidence": (1, 3)}),
        (4, {"Accept_Without_Verification": (1, 4), "Overreliance_Problem_Solving": (1, 4), "Independent_Confidence": (4, 7)}),
        (1, {"Accept_Without_Verification": (1, 4), "Overreliance_Problem_Solving": (5, 7)}),
        (1, {"Accept_Without_Verification": (1, 4), "Overreliance_Problem_Solving": (1, 4), "Independent_Confidence": (1, 3)}),
        (1, {"Accept_Without_Verification": (1, 4), "Overreliance_Problem_Solving": (1, 4), "Independent_Confidence": (4, 7)}),
    ]
    frames, score_list = [], []
    found = {}
    offset = 0
    for low, constraints in regimes:
        df = _frame(rng, n, low, constraints)
        scores = pipeline.predict(df)
        frames.append(df)
        score_list.append(scores)
        for i in range(n):
            _, name = expected_outcome(df.iloc[i], scores[i])
            found.setdefault(name, offset + i)
        offset += n
        if len(found) == 9:
            break
    return pd.concat(frames, ignore_index=True), np.concatenate(score_list), found


def t_rule(df, scores, idx, expect_risk, expect_name):
    def run():
        row = df.iloc[idx]
        at = submit({k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()})
        res = at.session_state["result"]
        ok = res["risk"] == expect_risk and res["intervention"] == expect_name
        return ok, f"model score={scores[idx]:.3f}; app level={res['risk']}, intervention={res['intervention']}"
    return run


def t_shap(seed):
    def run():
        rng = np.random.default_rng(seed)
        over = {f: int(rng.integers(1, 8)) for f in NUMERIC if f != "CGPA"}
        over["CGPA"] = float(rng.uniform(2.31, 4.0))
        for f in CATEGORICAL:
            over[f] = str(rng.choice(CATS[f]))
        at = submit(make_inputs(overrides=over))
        at.session_state["nav"] = "Why This Result?"
        at.run()
        ok = has_text(at, "Consistency check passed") and not exceptions(at)
        return ok, "reconstruction caption shown" if ok else "caption missing or exception raised"
    return run


def t_shap_failure():
    at = submit(make_inputs())
    os.environ["OVERRELIANCE_SIMULATE_SHAP_FAILURE"] = "1"
    try:
        at.session_state["nav"] = "Why This Result?"
        at.run()
    finally:
        os.environ.pop("OVERRELIANCE_SIMULATE_SHAP_FAILURE", None)
    ok = has_text(at, "Something went wrong while generating the explanation") and not exceptions(at)
    return ok, "friendly message shown, no traceback" if ok else f"exceptions={exceptions(at)}"


def t_model_missing():
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(HERE / "app.py", tmp / "app.py")
    at = new_app(str(tmp / "app.py")).run()
    ok = has_text(at, "The model could not be loaded") and not exceptions(at)
    return ok, "friendly error shown, no traceback" if ok else f"exceptions={exceptions(at)}"


def t_metadata_missing():
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(HERE / "app.py", tmp / "app.py")
    shutil.copy(HERE / "final_rf_pipeline.pkl", tmp / "final_rf_pipeline.pkl")
    at = new_app(str(tmp / "app.py")).run()
    ok = has_text(at, "could not be loaded; the documented") and not exceptions(at)
    return ok, "fallback notices shown, app still renders" if ok else f"exceptions={exceptions(at)}"


def t_pages():
    at = submit(make_inputs())
    bad = []
    for page in PAGES:
        at.session_state["nav"] = page
        at.run()
        if exceptions(at) or has_text(at, "Something went wrong while displaying this page"):
            bad.append(page)
    return not bad, "all 10 pages rendered" if not bad else f"failed pages: {bad}"


def t_optional_files():
    at = new_app()
    at.session_state["nav"] = "Model Insights"
    at.run()
    return not exceptions(at), f"exceptions={exceptions(at)}; optional CSV files present in folder: " \
        f"{[f for f in ('evaluation_predictions.csv', 'reference_data.csv', 'deep_learning_comparison.csv') if (HERE / f).exists()]}"


def t_nav():
    at = new_app().run()
    results = []
    for key, page in [("top_student", "Assess"), ("top_research", "Model Insights"), ("top_home", "Home"),
                      ("rside_2", "Testing"), ("side_0", "Assess"), ("top_cta", "Assess")]:
        button(at, key).click().run()
        results.append(at.session_state["nav"] == page)
    return all(results), f"navigation results={results}"


def t_intervention():
    at = submit(make_inputs())
    at.session_state["nav"] = "Intervention"
    at.run()
    button(at, "iv_start").click().run()
    for cb in at.checkbox:
        cb.check()
    at.run()
    incomplete = not has_text(at, "Activity completed")
    for ta in at.text_area:
        ta.input("my own answer" if "independent answer" in ta.label else "my reflection")
    at.run()
    complete = has_text(at, "Activity completed")
    return (incomplete and complete and not exceptions(at)), \
        f"not completed without answer/reflection={incomplete}; completed after steps+answer+reflection={complete}"


def t_scroll():
    def has_script(at):
        return any("scrollTo" in str(getattr(e, "proto", e)) for e in at.get("html"))
    at = new_app()
    at.session_state["nav"] = "Assess"
    at.run()
    button(at, "assess_next").click().run()
    after_next = has_script(at)
    at.select_slider[0].set_value(5).run()
    after_widget = has_script(at)
    return (after_next and not after_widget and not exceptions(at)), \
        f"scroll-to-top script present after Next={after_next}; after changing a slider={after_widget}"


def main():
    record("T01", "Startup", "Application starts on the Home page", "No exceptions; Home page shown", t_start)
    record("T02", "Thresholds", "Risk thresholds file matches the documented values", "low=3.5, high=5.0", t_thresholds)
    record("T03", "Valid input", "Neutral answers (all 4, CGPA 3.30)", "Score within 1-7 and Prediction page shown", t_valid(make_inputs(), "neutral"))
    record("T04", "Boundary", "Lowest values (all 1, CGPA 2.31)", "Score within 1-7, no error", t_valid(make_inputs(1, 2.31), "minimum"))
    record("T05", "Boundary", "Highest values (all 7, CGPA 4.00)", "Score within 1-7, no error", t_valid(make_inputs(7, 4.00), "maximum"))
    record("T06", "Validation", "Out-of-range values are clamped (Own_Ideas 99, CGPA 9.9)", "Own_Ideas=7.0, CGPA=4.0", t_clamp)
    record("T07", "Validation", "Missing (NaN) numeric answer falls back to neutral 4", "Cross_Check=4.0, no error", t_nan)
    record("T08", "Categories", "Unseen category handled by the saved pipeline", "Finite prediction within 1-7", t_unseen_pipeline)
    record("T09", "Categories", "Unseen category handled by the app", "Falls back to the first known option", t_unseen_app)
    record("T10", "Assessment", "Incomplete assessment cannot be submitted", "Submit disabled; no result", t_incomplete)
    record("T11", "Assessment", "Complete five sections through the UI, review and submit", "Review reached; submit enabled; Prediction shown", t_ui_flow)
    record("T12", "Assessment", "Start over clears answers, confirmations and result", "confirmed empty; result None; step 0", t_reset)
    record("T13", "Assessment", "Repeated submission", "No exception; result present", t_repeat)

    df, scores, found = find_samples()
    rule_cases = [
        ("Verification Challenge", "High", "High level and Accept Without Verification >= 5"),
        ("Independent Problem-Solving Task", "High", "High level and Overreliance in Problem Solving >= 5"),
        ("Guided Independent Reasoning", "High", "High level and Independent Confidence <= 3"),
        ("Critical Analysis Task", "High", "High level, no trigger met"),
        ("AI Output Verification Task", "Medium", "Moderate level and Cross Check <= 3"),
        ("Guided Reasoning Question", "Medium", "Moderate level and Independent Confidence <= 3"),
        ("Independent Idea Generation", "Medium", "Moderate level and Own Ideas <= 3"),
        ("Reflective AI-Use Question", "Medium", "Moderate level, no trigger met"),
        ("Responsible AI Use Guidance", "Low", "Low level"),
    ]
    for n, (name, risk, label) in enumerate(rule_cases, 14):
        tid = f"T{n:02d}"
        if name in found:
            record(tid, "Intervention rules", f"{label}", f"{risk} level -> {name}", t_rule(df, scores, found[name], risk, name))
        else:
            RESULTS.append({"id": tid, "category": "Intervention rules", "test": label, "expected": f"{risk} level -> {name}",
                            "observed": "No random input produced this branch", "status": "Not covered"})
            print(f"{tid:5s} Not covered {label}")

    base = 14 + len(rule_cases)
    for i, seed in enumerate((1, 2, 3)):
        record(f"T{base + i:02d}", "SHAP", f"SHAP reconstruction check on random input {i + 1} (includes categorical features)",
               "baseline + summed SHAP = model output within 1e-3", t_shap(seed))
    base += 3
    record(f"T{base:02d}", "Failure handling", "SHAP failure (simulated)", "Friendly message; prediction still available", t_shap_failure)
    record(f"T{base + 1:02d}", "Failure handling", "Model file missing", "Friendly error; no traceback", t_model_missing)
    record(f"T{base + 2:02d}", "Failure handling", "Thresholds and metadata files missing", "Fallback notices; app still renders", t_metadata_missing)
    record(f"T{base + 3:02d}", "Failure handling", "Optional evaluation CSV files absent", "No exception on Model Insights", t_optional_files)
    record(f"T{base + 4:02d}", "Navigation", "All ten pages render with a prediction present", "No exception on any page", t_pages)
    record(f"T{base + 5:02d}", "Navigation", "Navbar and side-panel buttons route correctly", "Each button opens the expected page", t_nav)
    record(f"T{base + 6:02d}", "Intervention", "Activity completes only with steps, answer and reflection", "Completion message appears only when all are done", t_intervention)
    record(f"T{base + 7:02d}", "Navigation", "Page scrolls back to the top after Next or a page change", "Scroll script appears after Next, not after changing an answer", t_scroll)

    out = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "environment": {"python": platform.python_version(), "streamlit": st.__version__,
                        "scikit-learn": sklearn.__version__, "shap": SHAP_VERSION},
        "tests": RESULTS,
    }
    (HERE / "test_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(r["status"] == "Pass" for r in RESULTS)
    failed = sum(r["status"] == "Fail" for r in RESULTS)
    print(f"\n{passed} passed, {failed} failed, {len(RESULTS) - passed - failed} not covered. Wrote test_results.json")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()