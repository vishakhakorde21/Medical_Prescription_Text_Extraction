# Handwritten Prescription → Structured Data Extraction

A vision-language model (VLM) pipeline that extracts structured patient and medication data from handwritten prescription images, using Google's Gemini API.

## Problem

Given a photo of a handwritten medical prescription, extract:
- Patient name, age
- Per medication: drug name, strength, frequency, duration

Input is noisy: no fixed template, doctor handwriting, medical abbreviations (TDS, OD, x1/52), varying image quality.

## Approach

**Chosen approach: Vision-Language Model (Gemini 2.5 Flash), prompt-only, no fine-tuning.**

### Why VLM over classical/cloud OCR

Classical OCR engines (Tesseract) and cloud handwriting OCR (Google Vision, AWS Textract) extract raw text but have no understanding of medical context - they cannot normalize "x 5/7" into a duration field, distinguish a drug name from a dosage instruction, or recognize that "TDS" and "Three times daily" mean the same thing. Handling that would require a separate rule-based or ML layer on top of OCR output anyway.

A VLM combines visual reading and semantic understanding in a single pass: it can be instructed (via prompt) to directly output a normalized, structured schema, apply domain knowledge (abbreviation expansion/normalization), and - critically - self-report uncertainty ("[UNCLEAR]") per field, which plain OCR cannot do.

**This tradeoff was tested, not assumed.** An OpenCV preprocessing pipeline (grayscale → Gaussian blur → adaptive threshold → denoising → morphological opening/closing — see `utils/image_processing.py`) was built specifically to prepare images for classical OCR. When the binarized output of this pipeline was fed to Gemini instead of the raw color image, extraction quality visibly dropped — the VLM relies on color, ink gradients, and anti-aliasing cues that thresholding destroys. This pipeline is retained in the repo as a documented negative result and as the preprocessing stage a Tesseract-based baseline would need, but it is **not used** in the final pipeline; Gemini receives the resized original color image.

### Why not a hybrid (OCR + LLM correction) or human-in-the-loop

Both are reasonable production directions (see [Scaling Strategy](#scaling-strategy-1k--100k-images) below) but were out of scope for a 6-image, single-day submission. The single-VLM-call approach was chosen to maximize what could be built, tested, and honestly evaluated within the time available.

## Architecture

```
images/*.jpg
     │
     ▼
image_processing.py  (load + resize only — color image preserved)
     │
     ▼
gemini_api.py  → Gemini 2.5 Flash
     │            - response_mime_type="application/json" (guarantees parseable JSON)
     │            - thinking_budget=0 (no internal reasoning needed for extraction; cuts latency/cost)
     │            - retry logic (3 attempts, 5s backoff) on transient server errors
     ▼
output/*.json  (per-prescription structured data)
     │
     ▼
accuracy_eval.py  → scored against ground_truth/*.json
```

## Setup

```bash
pip install -r requirements.txt
# Create .env with GOOGLE_API_KEY=your_key_here
python app.py                    # process all images in images/
python -m utils.accuracy_eval    # score against ground_truth/
```

`app.py` skips any image whose output JSON already exists in `output/`, to avoid re-spending API quota on reruns.

## Prompt Design

Full prompt: [`prompts/extraction_prompt.txt`](prompts/extraction_prompt.txt). Key design decisions, each arrived at through iteration against measured results (not guessed upfront):

- **Explicit conservative instruction for patient names.** Cursive patient names proved to be the single largest source of hallucination in early testing (6/6 images, model confidently guessing illegible names). Adding an explicit instruction to prefer `"[UNCLEAR]"` over a low-confidence guess for names specifically reduced hallucination rate by more than half, at the cost of the model now under-answering `patient_name` (visible in the results below) — an explicit, deliberate accuracy/safety tradeoff.
- **No forced abbreviation normalization.** An earlier version instructed the model to *always* return frequency as a short code (OD/BD/TDS). This backfired: on one prescription the doctor had literally written the word "day" (not an abbreviation), and forcing abbreviation output caused the model to fabricate a code that wasn't actually written. The final prompt instructs transcription of what is actually on the page, with abbreviation expansion suppressed only in the "don't turn OD into 'Once daily'" direction, not the reverse.
- **JSON mode over prompt-enforced formatting.** Gemini's `response_mime_type="application/json"` API parameter is used instead of prompt instructions like "do not use Markdown / triple backticks." This moved JSON-validity from a soft prompt request (which the model sometimes ignored) to a hard API-level guarantee — output was 100% parseable JSON across every run in this project, with zero manual markdown-stripping needed.

## Evaluation Methodology

**Ground truth** was hand-labeled by the author directly from the 6 source images, independently of any model output, to avoid anchoring bias. Fields genuinely illegible to a human reader (notably most patient names — fast cursive on 5-6 of 6 samples) were labeled `"[UNCLEAR]"` rather than guessed, so the ground truth itself reflects an honest legibility ceiling, not just a lookup table for "correct" values.

**Scoring** (`utils/accuracy_eval.py`) computes, per field:
- **Correct** — exact match after normalizing whitespace/case/common formatting variants (e.g. `"1g"` vs `"1 g"`, `"x 5/7"` vs `"5/7"` are treated as equivalent; genuine content differences are not)
- **Mismatch** — both sides have a real value, but they differ
- **Missed** — model returned `[UNCLEAR]`/null where a real value existed (safe failure)
- **Hallucination** — model returned a confident value where ground truth is `[UNCLEAR]` (i.e. no human could confidently read it either)

Fields are weighted per the problem statement: `drug_name`, `strength` (×3), `frequency`, `duration` (×2), `patient_name`, `age` (×1) — safety-critical fields count more toward the final score.

## Results

All 6 images, single model (`gemini-2.5-flash`), final locked prompt version.

| Metric | Value |
|---|---|
| Weighted field accuracy | **63.2%** |
| Hallucination rate | **5.9%** |
| Precision when confident (non-`[UNCLEAR]` answers) | **71%** |
| `[UNCLEAR]` justified rate | 33.3% |
| Avg. latency per image | ~5-7 sec |
| Schema validity | 100% (6/6 valid JSON, zero parse failures) |

### Per-image accuracy

| Image | Accuracy | Hallucination |
|---|---|---|
| prescription1 | 68.8% | 7.1% |
| prescription2 | 72.7% | 0% |
| prescription3 | 53.1% | 7.1% |
| prescription4 | 84.4% | 0% |
| prescription5 | 28.1% | 21.4% |
| prescription6 | 71.9% | 0% |

### Calibration

Calibration was measured as: (a) precision when the model gives a confident (non-`[UNCLEAR]`) answer, and (b) how often an `[UNCLEAR]` flag was actually justified (i.e. the field was genuinely unreadable to a human too), computed directly from existing predictions vs. ground truth — no additional API calls required.

- **71% precision when confident** — when the model commits to an answer, it is right roughly 7 times out of 10.
- **33.3% `[UNCLEAR]` justified rate** — of the 18 fields the model marked `[UNCLEAR]`, only 6 were genuinely illegible per the human-labeled ground truth. The remaining 12 (mostly `duration`) were fields a human could read but the model chose not to answer.

This is an honest calibration finding, not a purely positive one: the prompt succeeded at making the model *conservative* (low hallucination, 5.9%), but overshot on `duration` specifically, trading missed-but-readable answers for safety. Given the problem statement's explicit framing — "fabricating a dose is worse than leaving it blank" — this is judged to be the correct direction to err in, even though it costs raw accuracy points.

### Hallucination is concentrated on the hardest image

`prescription5` accounts for the largest share of hallucinations (21.4%) and the lowest accuracy (28.1%) in the set — its handwriting is visibly messier than the other 5 images even to a human reader. This suggests hallucination rate correlates with genuine image legibility rather than being a flat, random error rate — a reassuring signal for calibration, though only observable across 6 samples here.

## Cost

Gemini 2.5 Flash pricing: $0.30 / 1M input tokens, $2.50 / 1M output tokens.

Estimated per-image cost (resized ~800px image ≈ 250-350 image tokens + ~500-600 token prompt + ~150-250 token JSON output):

- **≈ $0.0004 - $0.0006 per image** (well under ₹0.05)
- **At 1,000 images: ≈ $0.40 - $0.60**
- **At 100,000 images: ≈ $40 - $60** (input/output token cost scales linearly; no fixed per-request fee on this API)

`thinking_budget=0` was set explicitly to avoid billing for internal reasoning tokens not needed for a direct extraction task — an early version without this setting had ~3x higher latency and proportionally higher output-token cost for no measurable accuracy gain.

## Performance

- **Latency**: ~5-7 sec/image typical (after disabling thinking mode and enabling JSON mode; an earlier version without these optimizations averaged ~14-15 sec/image — roughly a 2-3x improvement from these two changes alone).
- **Throughput under concurrent load**: **not implemented.** The current pipeline processes images strictly sequentially. At scale (1K+ images), this is a real bottleneck: 1,000 images × ~6 sec ≈ 100 minutes wall-clock, even though the API itself could handle concurrent requests. A production version would use `asyncio` or a thread pool with a concurrency limit respecting API rate limits, reducing wall-clock time roughly proportional to the concurrency level at unchanged per-image cost. Not built here due to time constraints on a same-day submission.
- **Schema consistency**: 100% valid, parseable JSON across every run — enforced structurally via `response_mime_type="application/json"`, not by prompt instruction alone (an earlier prompt-only approach occasionally returned markdown-wrapped JSON that required manual stripping and was less reliable).

## Robustness

| Edge case | Status |
|---|---|
| Illegible handwriting | Tested — demonstrated on prescription5 and on patient names generally; system responds with `[UNCLEAR]` rather than fabricating, at the accuracy cost discussed above |
| Non-standard abbreviations | Tested — prompt explicitly handles TD, PRN alongside OD/BD/TDS/QDS/HS/SOS, and instructs literal transcription (not forced normalization) when the doctor writes something outside this list |
| Cropped/rotated images | **Not tested** — all 6 source images were reasonably well-oriented full-page photos |
| Multiple languages/scripts | **Not tested** — all source prescriptions were in English |

The prompt-based approach makes no fixed-template assumptions, so it should generalize reasonably to cropped/rotated inputs and other languages, but this is unverified against this dataset and should not be assumed to hold without testing.

## Scaling Strategy (1K → 100,000 images)

Hand-labeling ground truth for every image, as done here for 6 images, does not scale. A production evaluation strategy would use:

1. **Stratified sampling for ground truth** — label a representative sample (~100-150 images, stratified across handwriting legibility, prescription format, and image quality) rather than the full dataset, to estimate accuracy within a statistical confidence interval.
2. **Multi-annotator agreement** on that sample, to establish a human noise floor — fields where independent human labelers disagree with each other set a realistic upper bound on what the model can be expected to get "right."
3. **No-ground-truth proxy signals** for the remaining unlabeled images: schema validity rate (already 100% here), `[UNCLEAR]`/null rate per field over time (a spike signals a new failure mode worth investigating), and self-consistency across repeated runs of the same image (low agreement across runs is a confidence proxy that requires no ground truth at all).
4. **Active learning / human-in-the-loop review**, prioritized by the above proxy signals — route the lowest-confidence ~10-15% of images to human review rather than reviewing everything, concentrating the added cost of human-in-the-loop review where it has the most value.

## Known Limitations

- `duration` is the field most often left `[UNCLEAR]` when a human could read it (12 of 18 `[UNCLEAR]` calls across the dataset) — a deliberate conservative tradeoff, not a bug, but one that measurably costs accuracy points under the current weighting.
- Small number of test images (6) limits statistical confidence in all reported metrics; per-image accuracy varies widely (28% to 84%), which is expected variance at this sample size rather than necessarily representative of population-level performance.
- No concurrent/batch processing implemented (see Performance section).
- Two Gemini model versions were used across development (`gemini-2.5-flash` for the primary pipeline; a brief detour to `gemini-3.5-flash` occurred mid-project when a newly created API project temporarily lost access to `gemini-2.5-flash` due to a Google-side model availability change for new projects). All final reported results use `gemini-2.5-flash` consistently across all 6 images.
- Calibration is measured via `[UNCLEAR]`-flag correctness rather than an explicit numeric confidence score, since the model does not natively output one; this is judged to be a reasonable proxy given the system design, but is not identical to true probabilistic calibration.

## Repository Structure

```
├── app.py                      # main batch processing entry point
├── app_single_image.py         # early single-image test script (superseded by app.py)
├── requirements.txt
├── .env                        # GOOGLE_API_KEY (excluded from git via .gitignore)
├── prompts/
│   └── extraction_prompt.txt   # final locked prompt
├── utils/
│   ├── image_processing.py     # OpenCV preprocessing (resize used; binarization pipeline
│   │                            #   retained for documentation, not used in final pipeline)
│   ├── gemini_api.py           # Gemini client + extraction call
│   ├── helper.py               # prompt file loader
│   ├── accuracy_eval.py        # scoring, detailed report, calibration analysis
├── images/                     # source prescription images
├── output/                     # model predictions (JSON) + intermediate preprocessing artifacts
└── groundtruth/               # hand-labeled ground truth (JSON)
```
