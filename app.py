import re
import io
import torch
import pytesseract
import streamlit as st
from PIL import Image
from pytesseract import Output
from transformers import AutoProcessor, AutoModelForTokenClassification

# ----------------------------------------------------------------------
# CONFIG — replace with your Hugging Face username / model repo name
# ----------------------------------------------------------------------
MODEL_REPO = "Dagem/receiptiq-layoutlmv3"

st.set_page_config(
    page_title="ReceiptIQ",
    page_icon="🧾",
    layout="wide",
)

# ----------------------------------------------------------------------
# STYLING — light custom CSS for a cleaner, more modern look
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .field-card {
        background-color: #f7f7f9;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid #e6e6e9;
    }
    .field-label {
        font-size: 0.8rem;
        color: #6b6b76;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .field-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a1f;
    }
    .confidence-badge {
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 999px;
        margin-left: 8px;
        vertical-align: middle;
    }
    .conf-high { background-color: #dcfce7; color: #166534; }
    .conf-low  { background-color: #fef3c7; color: #92400e; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# MODEL LOADING — cached so it only loads once per session
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_REPO, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_REPO)
    model.eval()
    label_list = list(model.config.id2label.values())
    return processor, model, label_list


@st.cache_resource(show_spinner=False)
def load_category_classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# ----------------------------------------------------------------------
# ENRICHMENT LOGIC (category / currency / tax) — same rules built and
# validated in the project notebook
# ----------------------------------------------------------------------
CANDIDATE_LABELS = ["Meals", "Travel", "Office Supplies", "Groceries", "Fuel", "Entertainment", "Other"]

KNOWN_VENDORS = {
    "MCDONALD": "Meals", "KFC": "Meals", "STARBUCKS": "Meals", "GERBANG ALAF": "Meals",
    "SHELL": "Fuel", "PETRONAS": "Fuel", "ESSO": "Fuel", "CALTEX": "Fuel",
    "MR D.I.Y": "Office Supplies", "IKEA": "Office Supplies", "POPULAR BOOK": "Office Supplies",
    "BOOK": "Office Supplies", "MACHINERY": "Office Supplies",
    "GIFT": "Entertainment", "DECO": "Entertainment",
}

KEYWORD_RULES = {
    "RESTAURANT": "Meals", "RESTORAN": "Meals", "CAFE": "Meals", "KOPITIAM": "Meals",
    "SEAFOOD": "Meals", "CHICKEN": "Meals", "BAKERY": "Meals", "CAKE": "Meals",
    "CONFECTIONERY": "Meals",
    "MART": "Groceries", "GROCER": "Groceries", "SUPERMARKET": "Groceries",
    "STATIONERY": "Office Supplies", "STATIONERS": "Office Supplies",
    "HARDWARE": "Office Supplies", "PRINT": "Office Supplies",
    "PETROL": "Fuel", "STATION": "Fuel",
    "FLORIST": "Entertainment", "HANDICRAFT": "Entertainment", "GIFT": "Entertainment",
}

TAX_KEYWORDS = r"\bGST\b|\bTAX\b|\bVAT\b|\bSST\b"
CURRENCY_PATTERNS = {
    "MYR": r"\bRM\b|\bMYR\b",
    "USD": r"\bUSD\b",
    "SGD": r"\bSGD\b",
    "EUR": r"\bEUR\b",
    "GBP": r"\bGBP\b",
}


def classify_category(vendor_name, classifier):
    if not vendor_name or vendor_name.strip() == "":
        return "Other", 0.0

    upper_name = vendor_name.upper()

    for key, category in KNOWN_VENDORS.items():
        if key in upper_name:
            return category, 1.0

    for key, category in KEYWORD_RULES.items():
        if key in upper_name:
            return category, 0.9

    result = classifier(f"Receipt from vendor: {vendor_name}", CANDIDATE_LABELS)
    category, score = result["labels"][0], result["scores"][0]

    if score < 0.4:
        return "Uncertain", score
    return category, score


def detect_currency(words):
    full_text = " ".join(words).upper()
    for currency, pattern in CURRENCY_PATTERNS.items():
        if re.search(pattern, full_text):
            return currency
    return "Unknown"


def detect_tax(words):
    upper_words = [w.upper() for w in words]
    full_text = " ".join(words)
    has_tax_mention = bool(re.search(TAX_KEYWORDS, full_text.upper()))

    tax_rate = None
    for i, w in enumerate(upper_words):
        if re.search(TAX_KEYWORDS, w):
            window = " ".join(words[max(0, i - 1):i + 4])
            rate_match = re.search(r"(\d{1,2})\s*%", window)
            if rate_match:
                tax_rate = f"{int(rate_match.group(1))}%"
                break
    return has_tax_mention, tax_rate


# ----------------------------------------------------------------------
# OCR + INFERENCE PIPELINE
# ----------------------------------------------------------------------
def run_ocr(image):
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    width, height = image.size
    words, boxes = [], []

    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        if word == "":
            continue
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        box = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]
        words.append(word)
        boxes.append(box)
    return words, boxes


def run_inference(image, words, boxes, processor, model, label_list):
    encoding = processor(
        image, words, boxes=boxes,
        truncation=True, padding="max_length", return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**encoding)

    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    seen = set()
    tagged = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx in seen:
            continue
        seen.add(word_idx)
        tagged.append((words[word_idx], label_list[predictions[idx]]))
    return tagged


def collect_fields(tagged_words):
    fields = {"S-COMPANY": [], "S-DATE": [], "S-ADDRESS": [], "S-TOTAL": []}
    for word, label in tagged_words:
        if label in fields:
            fields[label].append(word)
    return {
        "vendor": " ".join(fields["S-COMPANY"]) or "Not found",
        "date": " ".join(fields["S-DATE"]) or "Not found",
        "address": " ".join(fields["S-ADDRESS"]) or "Not found",
        "total": " ".join(fields["S-TOTAL"]) or "Not found",
    }


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🧾 ReceiptIQ")
st.caption("Upload a receipt photo — extraction powered by a fine-tuned LayoutLMv3 model.")

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.image(image, caption="Uploaded receipt", use_container_width=True)

    with st.spinner("Loading model..."):
        processor, model, label_list = load_model()

    with st.spinner("Reading receipt..."):
        words, boxes = run_ocr(image)

    if len(words) == 0:
        st.error("No readable text found in this image. Try a clearer photo.")
    else:
        with st.spinner("Extracting fields..."):
            tagged_words = run_inference(image, words, boxes, processor, model, label_list)
            fields = collect_fields(tagged_words)

        with st.spinner("Classifying category, currency, and tax..."):
            classifier = load_category_classifier()
            category, confidence = classify_category(fields["vendor"], classifier)
            currency = detect_currency(words)
            has_tax, tax_rate = detect_tax(words)

        with col2:
            st.subheader("Extracted fields")

            def field_card(label, value, badge=None):
                badge_html = f'<span class="confidence-badge {badge[1]}">{badge[0]}</span>' if badge else ""
                st.markdown(
                    f"""
                    <div class="field-card">
                        <div class="field-label">{label}</div>
                        <div class="field-value">{value}{badge_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            field_card("Vendor", fields["vendor"])
            field_card("Date", fields["date"])
            field_card("Total", fields["total"])
            field_card("Address", fields["address"])

            conf_class = "conf-high" if confidence >= 0.7 else "conf-low"
            field_card("Category", category, badge=(f"{confidence:.0%} confidence", conf_class))
            field_card("Currency", currency)
            field_card("Tax mentioned", "Yes" if has_tax else "No", badge=(tax_rate, "conf-high") if tax_rate else None)

        with st.expander("Raw OCR words (debug view)"):
            st.write(words)

else:
    st.info("Upload a receipt image to get started.")
