"""
Scam Detection & Recommendation System — Streamlit UI
======================================================
Full workflow:
  1. User pastes a product URL
  2. UI calls backend /scrape  → gets listing data
  3. UI calls backend /scan    → gets scam score + breakdown
  4. UI displays gauge, risk factors, model breakdown
  5. UI shows trusted platform recommendations

Run backend first:
  cd Backend && uvicorn main:app --reload

Then run UI:
  streamlit run app.py
"""

import re
import math
import streamlit as st
import requests
from urllib.parse import quote_plus

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://127.0.0.1:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scam Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .risk-box {
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 1.1rem;
        margin: 10px 0;
    }
    .safe    { background-color: #1a3a1a; border: 2px solid #2ecc71; color: #2ecc71; }
    .medium  { background-color: #3a2e10; border: 2px solid #f39c12; color: #f39c12; }
    .danger  { background-color: #3a1010; border: 2px solid #e74c3c; color: #e74c3c; }
    .reason-box {
        background-color: #1e1e2e;
        border-left: 4px solid #e74c3c;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
    .ok-box {
        background-color: #1e1e2e;
        border-left: 4px solid #2ecc71;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
        font-size: 0.95rem;
    }
    .rec-button {
        display: inline-block;
        background-color: #1a73e8;
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        text-decoration: none !important;
        margin: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .rec-button:hover { background-color: #1558b0; }
    .breakdown-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 10px 16px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .backend-error {
        background-color: #3a1010;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 14px;
        color: #e74c3c;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Backend helpers ───────────────────────────────────────────────────────────
def check_backend() -> bool:
    """Returns True if the FastAPI backend is reachable."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def api_scrape(url: str) -> dict:
    """Call /scrape endpoint. Returns dict or raises on failure."""
    r = requests.post(f"{API_BASE}/scrape", json={"url": url}, timeout=20)
    r.raise_for_status()
    return r.json()


def api_scan(product: dict, fusion: str) -> dict:
    """Call /scan endpoint. Returns dict or raises on failure."""
    # Clean price — backend expects float or None, scraper may return string
    raw_price = product.get("price")
    try:
        price = float(str(raw_price).replace(",", "")) if raw_price else None
    except Exception:
        price = None

    # Clean seller_rating similarly
    raw_rating = product.get("seller_rating")
    try:
        seller_rating = float(raw_rating) if raw_rating else None
    except Exception:
        seller_rating = None

    payload = {
        "title":         str(product.get("title") or ""),
        "description":   str(product.get("description") or ""),
        "price":         price,
        "phone_model":   str(product.get("phone_model") or "unknown"),
        "image_path":    "",
        "seller_rating": seller_rating,
        "fusion":        fusion,
    }
    r = requests.post(f"{API_BASE}/scan", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def api_recommend() -> list:
    """Call /recommend endpoint."""
    r = requests.get(f"{API_BASE}/recommend", timeout=5)
    r.raise_for_status()
    return r.json().get("recommendations", [])


# ── Gauge chart ───────────────────────────────────────────────────────────────
def render_gauge(score: float):
    pct = score * 100
    if pct < 40:
        color     = "#2ecc71"
        label     = "LOW RISK"
        css_class = "safe"
    elif pct < 65:
        color     = "#f39c12"
        label     = "MEDIUM RISK"
        css_class = "medium"
    else:
        color     = "#e74c3c"
        label     = "HIGH RISK"
        css_class = "danger"

    cx, cy, r_arc = 100, 90, 70
    start_angle   = 180
    end_angle     = 180 + (pct / 100) * 180
    x1 = cx + r_arc * math.cos(math.radians(start_angle))
    y1 = cy + r_arc * math.sin(math.radians(start_angle))
    x2 = cx + r_arc * math.cos(math.radians(end_angle))
    y2 = cy + r_arc * math.sin(math.radians(end_angle))
    large = 1 if (end_angle - start_angle) > 180 else 0

    svg = f"""
    <svg viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg">
      <path d="M 30 90 A 70 70 0 0 1 170 90"
            fill="none" stroke="#2a2a3e" stroke-width="16" stroke-linecap="round"/>
      <path d="M {x1:.1f} {y1:.1f} A 70 70 0 {large} 1 {x2:.1f} {y2:.1f}"
            fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
      <text x="100" y="85" text-anchor="middle" font-size="26"
            font-weight="bold" fill="{color}">{int(pct)}%</text>
      <text x="100" y="105" text-anchor="middle" font-size="11" fill="#888">{label}</text>
      <text x="28"  y="108" text-anchor="middle" font-size="9" fill="#555">0%</text>
      <text x="172" y="108" text-anchor="middle" font-size="9" fill="#555">100%</text>
    </svg>
    """
    st.markdown(f'<div style="text-align:center">{svg}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="risk-box {css_class}"><b>{label}</b> — Scam probability: {int(pct)}%</div>',
        unsafe_allow_html=True,
    )


# ── Risk reasons ──────────────────────────────────────────────────────────────
def get_reasons(scan_result: dict, product: dict) -> list[tuple[str, bool]]:
    """
    Returns list of (reason_text, is_danger) tuples.
    is_danger=True → red box, False → green box.
    """
    reasons = []
    scores  = scan_result.get("scores", {})

    # Text signal
    text_score = scores.get("lstm", scores.get("tfidf", 0))
    if text_score > 0.6:
        reasons.append(("🔤 Suspicious language detected in title or description", True))
    elif text_score < 0.3:
        reasons.append(("🔤 Listing language looks normal and trustworthy", False))

    # Image signal
    img_score = scores.get("resnet50", scores.get("efficientnet", 0))
    if img_score > 0.6:
        reasons.append(("🖼️ Product image shows signs associated with scam listings", True))
    elif img_score < 0.3:
        reasons.append(("🖼️ Product image appears consistent with trusted listings", False))

    # Tabular signal
    ml_score = scores.get("xgboost", scores.get("random_forest", 0))
    if ml_score > 0.6:
        reasons.append(("💰 Price or seller details match known scam patterns", True))
    elif ml_score < 0.3:
        reasons.append(("💰 Price and seller details look reasonable", False))

    # Missing fields
    if not product.get("description"):
        reasons.append(("📋 No product description provided — common in scam listings", True))
    if not product.get("price"):
        reasons.append(("❓ Price is missing or could not be detected", True))

    if not reasons:
        reasons.append(("✅ No major red flags detected", False))

    return reasons


# ── Model breakdown ───────────────────────────────────────────────────────────
def render_breakdown(scan_result: dict):
    scores  = scan_result.get("scores", {})
    weights = scan_result.get("weights", {})

    MODEL_LABELS = {
        "lstm":          "LSTM (Text)",
        "tfidf":         "TF-IDF (Text)",
        "resnet50":      "ResNet50 (Image)",
        "efficientnet":  "EfficientNet (Image)",
        "xgboost":       "XGBoost (Tabular)",
        "random_forest": "Random Forest (Tabular)",
    }

    for model_name, score in scores.items():
        label  = MODEL_LABELS.get(model_name, model_name)
        weight = weights.get(model_name, 0)
        pct    = int(score * 100)
        bar_w  = max(4, pct)
        color  = "#2ecc71" if pct < 40 else "#f39c12" if pct < 65 else "#e74c3c"

        st.markdown(f"""
        <div class="breakdown-row">
            <span style="width:180px">{label}</span>
            <div style="flex:1; margin:0 12px; background:#2a2a3e; border-radius:4px; height:10px;">
                <div style="width:{bar_w}%; background:{color}; height:10px; border-radius:4px;"></div>
            </div>
            <span style="width:40px; text-align:right; color:{color}">{pct}%</span>
            <span style="width:60px; text-align:right; color:#555; font-size:0.8rem">w={int(weight*100)}%</span>
        </div>
        """, unsafe_allow_html=True)


# ── Recommendations ───────────────────────────────────────────────────────────
TRUSTED_SITES = {
    "🛒 Amazon Egypt": "https://www.amazon.eg/s?k={query}&i=electronics",
    "📦 Jumia Egypt":  "https://www.jumia.com.eg/catalog/?q={query}",
    "🌙 Noon Egypt":   "https://www.noon.com/egypt-en/search/?q={query}",
}

def render_recommendations(title: str):
    query   = re.sub(r"[^\w\s]", "", title or "smartphone")
    query   = " ".join(query.split()[:5])
    encoded = quote_plus(query)

    st.markdown("### 🛡️ Find it from a trusted source")
    st.markdown("Buy the same product safely from these verified platforms:")

    cols = st.columns(len(TRUSTED_SITES))
    for col, (name, url_template) in zip(cols, TRUSTED_SITES.items()):
        link = url_template.format(query=encoded)
        col.markdown(
            f'<a href="{link}" target="_blank" class="rec-button">{name}</a>',
            unsafe_allow_html=True,
        )


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🔍 Scam Detection System")
    st.markdown("Paste a product listing URL to check if it's safe to buy.")

    # ── Backend status ──
    backend_ok = check_backend()
    if not backend_ok:
        st.markdown("""
        <div class="backend-error">
        ⚠️ <b>Backend is not running.</b><br>
        Start it first:<br>
        <code>cd Backend &amp;&amp; uvicorn main:app --reload</code>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.success("✅ Backend connected", icon="🟢")

    # ── Sidebar — settings ──
    with st.sidebar:
        st.header("⚙️ Settings")
        fusion_choice = st.radio(
            "Fusion Model",
            options=["A", "B"],
            format_func=lambda x: (
                "Sequence A — LSTM + ResNet50 + XGBoost"
                if x == "A"
                else "Sequence B — TF-IDF + EfficientNet + Random Forest"
            ),
            index=0,
            help="Choose which combination of models to use for analysis.",
        )
        st.markdown("---")
        st.markdown("**Sequence A** (recommended)")
        st.markdown("• LSTM 40% · ResNet50 35% · XGBoost 25%")
        st.markdown("**Sequence B**")
        st.markdown("• TF-IDF 40% · EfficientNet 35% · Random Forest 25%")

    # ── URL input ──
    url = st.text_input(
        "Product URL",
        placeholder="https://www.dubizzle.com.eg/en/ad/iphone-15-pro-...",
    )

    analyze_btn = st.button("🔍 Analyze Product", type="primary", use_container_width=True)

    if analyze_btn and not url:
        st.warning("Please enter a product URL first.")
        return

    if analyze_btn and url:
        if not url.startswith("http"):
            st.error("Please enter a valid URL starting with http:// or https://")
            return

        # ── Step 1: Scrape ──
        with st.spinner("🔎 Fetching product details from URL..."):
            try:
                product = api_scrape(url)
            except Exception as e:
                st.error(f"Scraping failed: {e}")
                return

        # Product card
        st.markdown("---")
        st.markdown("### 📱 Product Details")
        col1, col2 = st.columns([1, 2])
        with col1:
            if product.get("image_url"):
                st.image(product["image_url"], use_container_width=True)
            else:
                st.markdown("🖼️ *No image found*")
        with col2:
            st.markdown(f"**{product.get('title') or 'Unknown Title'}**")
            if product.get("price"):
                st.markdown(f"💵 **Price:** {product['price']} EGP")
            else:
                st.markdown("💵 **Price:** Not found")
            if product.get("description"):
                st.markdown(f"📋 {str(product['description'])[:200]}...")
            if product.get("source"):
                st.markdown(f"🌐 **Source:** {product['source']}")

        # ── Step 2: Scan ──
        st.markdown("---")
        st.markdown(f"### 🤖 Running AI Analysis (Fusion {fusion_choice})...")
        with st.spinner("Analyzing with text, image, and tabular models..."):
            try:
                scan_result = api_scan(product, fusion_choice)
            except Exception as e:
                st.error(f"Model analysis failed: {e}")
                return

        # ── Step 3: Gauge ──
        st.markdown("### 📊 Risk Score")
        render_gauge(scan_result["final_score"])

        # Verdict badge
        verdict    = scan_result["verdict"]
        risk_level = scan_result["risk_level"]
        verdict_color = {"Trusted": "#2ecc71", "Suspicious": "#f39c12", "Scam": "#e74c3c"}.get(verdict, "#888")
        st.markdown(
            f'<div style="text-align:center; font-size:1.3rem; font-weight:bold; color:{verdict_color}; margin:8px 0">'
            f'Verdict: {verdict}</div>',
            unsafe_allow_html=True,
        )

        # ── Step 4: Risk reasons (only shown when score > 59%) ──
        if scan_result["final_score"] > 0.59:
            st.markdown("### ⚠️ Risk Factors")
            reasons = get_reasons(scan_result, product)
            for reason_text, is_danger in reasons:
                css = "reason-box" if is_danger else "ok-box"
                st.markdown(f'<div class="{css}">{reason_text}</div>', unsafe_allow_html=True)

        # ── Step 6: Recommendations ──
        st.markdown("---")
        render_recommendations(product.get("title", ""))

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.8rem'>"
        "Scam Detection System — HNU Deep Learning Project 2025"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
