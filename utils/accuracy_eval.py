import json
import os
import re
import difflib


# Safety-critical fields weighted higher, per problem statement
FIELD_WEIGHTS = {
    "drug_name": 3,
    "strength": 3,
    "frequency": 2,
    "duration": 2,
    "patient_name": 1,
    "age": 1,
}


def normalize(value):
    """
    Normalizes a value so formatting differences (spacing, case,
    common prefixes/suffixes) don't count as errors.
    """
    if value is None:
        return None

    text = str(value).strip().lower()

    # Remove a leading "x " (e.g. "x 5/7" -> "5/7")
    text = re.sub(r"^x\s*", "", text)

    # Remove all internal whitespace (e.g. "1 g" -> "1g", "15 ml" -> "15ml")
    text = re.sub(r"\s+", "", text)

    # Normalize common age-unit variants (e.g. "28 YO" -> "28")
    text = text.replace("y/o", "").replace("yo", "")

    return text


def char_error_rate(pred, truth):
    """
    How different two strings are, from 0.0 (identical) to 1.0 (completely different),
    computed on normalized text so formatting doesn't inflate the error.
    """
    if pred is None or truth is None:
        return None

    pred_norm = normalize(pred)
    truth_norm = normalize(truth)

    if truth_norm == "":
        return None

    matcher = difflib.SequenceMatcher(None, pred_norm, truth_norm)
    similarity = matcher.ratio()

    return round(1 - similarity, 3)


def compare_field(pred_value, truth_value):
    """
    Classifies one field's prediction against ground truth.
    """

    pred_is_unclear = pred_value in (None, "[UNCLEAR]", "")
    truth_is_unclear = truth_value in (None, "[UNCLEAR]", "")

    if truth_is_unclear and pred_is_unclear:
        return {"status": "correct_unclear", "cer": None}

    if truth_is_unclear and not pred_is_unclear:
        return {"status": "hallucination", "cer": None}

    if not truth_is_unclear and pred_is_unclear:
        return {"status": "missed", "cer": None}

    # Exact match after normalization = correct, even if raw strings differ
    if normalize(pred_value) == normalize(truth_value):
        return {"status": "correct", "cer": 0.0}

    cer = char_error_rate(pred_value, truth_value)
    status = "correct" if cer == 0 else "mismatch"

    return {"status": status, "cer": cer}


def score_prescription(prediction, truth):

    results = {}
    total_weight = 0
    earned_weight = 0

    for field in ("patient_name", "age"):
        result = compare_field(prediction.get(field), truth.get(field))
        results[field] = result
        weight = FIELD_WEIGHTS[field]
        total_weight += weight
        if result["status"] in ("correct", "correct_unclear"):
            earned_weight += weight

    pred_meds = prediction.get("medications", [])
    truth_meds = truth.get("medications", [])

    med_results = []
    max_len = max(len(pred_meds), len(truth_meds))

    for i in range(max_len):
        pred_med = pred_meds[i] if i < len(pred_meds) else {}
        truth_med = truth_meds[i] if i < len(truth_meds) else {}

        med_field_results = {}

        for field in ("drug_name", "strength", "frequency", "duration"):
            result = compare_field(pred_med.get(field), truth_med.get(field))
            med_field_results[field] = result
            weight = FIELD_WEIGHTS[field]
            total_weight += weight
            if result["status"] in ("correct", "correct_unclear"):
                earned_weight += weight

        med_results.append(med_field_results)

    results["medications"] = med_results

    weighted_accuracy = round(earned_weight / total_weight, 3) if total_weight else 0

    all_field_results = [results[f] for f in ("patient_name", "age")]
    for med in med_results:
        all_field_results.extend(med.values())

    hallucinations = sum(1 for r in all_field_results if r["status"] == "hallucination")
    hallucination_rate = round(hallucinations / len(all_field_results), 3)

    return {
        "weighted_accuracy": weighted_accuracy,
        "hallucination_rate": hallucination_rate,
        "field_results": results,
    }


def run_evaluation(output_dir="output", truth_dir="ground_truth"):

    scores = []

    for filename in sorted(os.listdir(truth_dir)):
        if not filename.endswith(".json"):
            continue

        truth_path = os.path.join(truth_dir, filename)
        pred_path = os.path.join(output_dir, filename)

        if not os.path.exists(pred_path):
            print(f"Missing prediction for {filename}, skipping.")
            continue

        with open(truth_path, "r", encoding="utf-8") as f:
            truth = json.load(f)

        with open(pred_path, "r", encoding="utf-8") as f:
            prediction = json.load(f)

        result = score_prescription(prediction, truth)
        result["file"] = filename
        scores.append(result)

        print(f"{filename}: accuracy={result['weighted_accuracy']}  "
              f"hallucination_rate={result['hallucination_rate']}")

    if scores:
        avg_accuracy = round(sum(s["weighted_accuracy"] for s in scores) / len(scores), 3)
        avg_hallucination = round(sum(s["hallucination_rate"] for s in scores) / len(scores), 3)

        print("\n" + "=" * 50)
        print(f"Average weighted accuracy   : {avg_accuracy}")
        print(f"Average hallucination rate  : {avg_hallucination}")
        print("=" * 50)

    return scores


def print_detailed_report(output_dir="output", truth_dir="ground_truth"):
    """
    Prints every field where prediction != ground truth,
    showing both values side by side, so you can see WHY
    a file scored low.
    """

    for filename in sorted(os.listdir(truth_dir)):
        if not filename.endswith(".json"):
            continue

        truth_path = os.path.join(truth_dir, filename)
        pred_path = os.path.join(output_dir, filename)

        if not os.path.exists(pred_path):
            continue

        with open(truth_path, "r", encoding="utf-8") as f:
            truth = json.load(f)

        with open(pred_path, "r", encoding="utf-8") as f:
            prediction = json.load(f)

        print(f"\n{'=' * 60}")
        print(f"{filename}")
        print("=" * 60)

        for field in ("patient_name", "age"):
            result = compare_field(prediction.get(field), truth.get(field))
            if result["status"] not in ("correct", "correct_unclear"):
                print(f"  [{result['status'].upper()}] {field}")
                print(f"      predicted : {prediction.get(field)!r}")
                print(f"      truth     : {truth.get(field)!r}")

        pred_meds = prediction.get("medications", [])
        truth_meds = truth.get("medications", [])
        max_len = max(len(pred_meds), len(truth_meds))

        for i in range(max_len):
            pred_med = pred_meds[i] if i < len(pred_meds) else {}
            truth_med = truth_meds[i] if i < len(truth_meds) else {}

            for field in ("drug_name", "strength", "frequency", "duration"):
                result = compare_field(pred_med.get(field), truth_med.get(field))
                if result["status"] not in ("correct", "correct_unclear"):
                    print(f"  [{result['status'].upper()}] medication[{i}].{field}")
                    print(f"      predicted : {pred_med.get(field)!r}")
                    print(f"      truth     : {truth_med.get(field)!r}")

def calibration_analysis(output_dir="output", truth_dir="ground_truth"):
    """
    Measures whether the model's confidence (giving a real value vs
    "[UNCLEAR]"/null) correlates with actual correctness.

    Uses only existing predictions and ground truth — no new API calls.
    """

    confident_correct = 0
    confident_wrong = 0
    unclear_correct = 0   # model said unclear, truth was also unclear -> good call
    unclear_missed = 0    # model said unclear, truth had a real value -> overcautious

    for filename in sorted(os.listdir(truth_dir)):
        if not filename.endswith(".json"):
            continue

        truth_path = os.path.join(truth_dir, filename)
        pred_path = os.path.join(output_dir, filename)

        if not os.path.exists(pred_path):
            continue

        with open(truth_path, "r", encoding="utf-8") as f:
            truth = json.load(f)

        with open(pred_path, "r", encoding="utf-8") as f:
            prediction = json.load(f)

        all_pairs = []

        for field in ("patient_name", "age"):
            all_pairs.append((prediction.get(field), truth.get(field)))

        pred_meds = prediction.get("medications", [])
        truth_meds = truth.get("medications", [])
        max_len = max(len(pred_meds), len(truth_meds))

        for i in range(max_len):
            pred_med = pred_meds[i] if i < len(pred_meds) else {}
            truth_med = truth_meds[i] if i < len(truth_meds) else {}

            for field in ("drug_name", "strength", "frequency", "duration"):
                all_pairs.append((pred_med.get(field), truth_med.get(field)))

        for pred_value, truth_value in all_pairs:
            result = compare_field(pred_value, truth_value)

            if result["status"] == "correct":
                confident_correct += 1
            elif result["status"] in ("mismatch", "hallucination"):
                confident_wrong += 1
            elif result["status"] == "correct_unclear":
                unclear_correct += 1
            elif result["status"] == "missed":
                unclear_missed += 1

    total_confident = confident_correct + confident_wrong
    precision_when_confident = (
        round(confident_correct / total_confident, 3) if total_confident else None
    )

    total_unclear_calls = unclear_correct + unclear_missed
    unclear_justified_rate = (
        round(unclear_correct / total_unclear_calls, 3) if total_unclear_calls else None
    )

    print("\n" + "=" * 50)
    print("CALIBRATION ANALYSIS")
    print("=" * 50)
    print(f"Confident predictions (real value given) : {total_confident}")
    print(f"  - Correct                              : {confident_correct}")
    print(f"  - Wrong (mismatch or hallucination)    : {confident_wrong}")
    print(f"  Precision when confident                : {precision_when_confident}")
    print()
    print(f"'[UNCLEAR]' calls made by model           : {total_unclear_calls}")
    print(f"  - Justified (truth was also unclear)   : {unclear_correct}")
    print(f"  - Overcautious (truth had real value)  : {unclear_missed}")
    print(f"  '[UNCLEAR]' justified rate              : {unclear_justified_rate}")
    print("=" * 50)

    return {
        "precision_when_confident": precision_when_confident,
        "unclear_justified_rate": unclear_justified_rate,
        "confident_correct": confident_correct,
        "confident_wrong": confident_wrong,
        "unclear_correct": unclear_correct,
        "unclear_missed": unclear_missed,
    }


if __name__ == "__main__":
    run_evaluation()
    print_detailed_report()
    calibration_analysis()