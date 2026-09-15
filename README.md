# ReceiptIQ

An interactive Streamlit app that extracts structured data from photographed
receipts — vendor, date, total, tax, currency, and spending category —
using a fine-tuned document-understanding model, without any manual data
entry.

## Problem

Manually entering receipt data (for expense reports, budgeting, or
bookkeeping) is tedious and error-prone. Generic OCR alone isn't enough,
since it returns raw text without knowing *which* text is the vendor name
versus the date versus the total. ReceiptIQ combines OCR with a model that
understands the *layout* of a receipt to correctly label each piece of text.

## Data

Fine-tuned on the **SROIE** dataset (Scanned Receipts OCR and Information
Extraction) — a public benchmark dataset of real scanned receipts with
labeled fields (company, date, address, total).

## Methodology

1. **OCR** — Tesseract extracts raw text and word-level bounding boxes from
   the uploaded receipt image.
2. **Field extraction** — a **LayoutLMv3** model, fine-tuned on SROIE,
   performs token classification over the OCR output: it doesn't just read
   the text, it uses each word's position on the page to decide whether
   it's a vendor name, date, total, etc.
3. **Category classification** — a two-stage approach: known
   vendors/keywords are matched first via a rule-based layer; if nothing
   matches, a general-purpose zero-shot classification model
   (`facebook/bart-large-mnli`) assigns a spending category.
4. **Currency & tax detection** — regex-based parsing over the extracted
   total/line-item text.
5. **Interface** — a Streamlit app displays each extracted field with a
   confidence indicator, and falls back to a clear "Not found" rather than
   guessing when a field can't be confidently extracted.

## Tech stack

Python, PyTorch, Hugging Face Transformers (LayoutLMv3), Tesseract OCR
(via pytesseract), Streamlit.

## Results

Validated through manual testing on real receipt photos rather than a
formal held-out benchmark — extraction quality holds up well in practice on
typical printed receipts. *(If you have quantitative results from the SROIE
fine-tuning run — e.g. token-level F1 per field — worth adding those here
alongside the manual testing note.)*

## Limitations

- Extraction quality depends on photo quality (lighting, angle, crease,
  blur) since it starts from OCR.
- The zero-shot category fallback is a general-purpose model, not
  fine-tuned for receipts, so it's less reliable than the rule-based match
  or the core field-extraction model.
- Fine-tuned specifically on SROIE-style receipts; unusual receipt formats
  (e.g. handwritten, non-English, or highly non-standard layouts) may not
  extract as reliably.
- Heavier resource footprint than a typical Streamlit app (PyTorch +
  Transformers + two loaded models) — first load can be slower due to
  downloading model weights from Hugging Face Hub.


## Future improvements

- Fine-tune category classification instead of relying on zero-shot
- Multi-language receipt support
- Quantitative evaluation report (per-field precision/recall)

## Run locally

1. Install Tesseract OCR (needed by `pytesseract`):
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki

2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open `app.py` and replace `MODEL_REPO` with your Hugging Face model repo
   (e.g. `"your-username/receiptiq-layoutlmv3"`).

4. Run the app:
   ```
   streamlit run app.py
   ```

5. Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Notes

- First load will download the model from Hugging Face Hub — may take a
  minute depending on connection speed.
- Category classification uses zero-shot classification (`facebook/bart-large-mnli`)
  with a rule-based layer for known vendors/keywords first.
