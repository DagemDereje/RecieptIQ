# ReceiptIQ

Upload a receipt photo and get structured fields back: vendor, date, total,
category, currency, and tax detection — powered by a LayoutLMv3 model
fine-tuned on the SROIE dataset.

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
