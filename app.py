"""
Scam Detection & Recommendation System — Streamlit UI
======================================================
Paste a product URL → scrape it → run through models → show risk score + recommendations.

NOTE: Model scores are currently DUMMY values (random). Swap in real model calls
      in the `run_models()` function once each model is ready.
"""

import re
import random
import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

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
    .product-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Scraper ───────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

def scrape_product(url: str) -> dict:
    """
    Scrape title, price, description, and first image from a product URL.
    Supports: Jumia, Amazon EG, OLX/Dubizzle, Noon, generic fallback.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = 'utf-8'
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return {"error": str(e)}

    title, price, description, image_url = "", "", "", ""

    # ── Jumia ──
    if "jumia.com" in url:
        title_el = soup.select_one("h1.-fs20.-pts.-pbxs")
        price_el = soup.select_one(".-b.-ltr.-tal.-fs24")
        desc_el  = soup.select_one(".-mhm.-pvl.-mod.article")
        img_el   = soup.select_one("img.-fw.-fh")

        title       = title_el.get_text(strip=True) if title_el else ""
        price       = _extract_price(price_el.get_text() if price_el else "")
        description = desc_el.get_text(strip=True)[:300] if desc_el else ""
        image_url   = img_el.get("data-src", img_el.get("src", "")) if img_el else ""

    # ── Amazon EG ──
    elif "amazon.eg" in url or "amazon.com" in url:
        title_el = soup.select_one("#productTitle")
        price_el = soup.select_one(".a-price-whole") or soup.select_one(".a-offscreen")
        desc_el  = soup.select_one("#feature-bullets")
        img_el   = soup.select_one("#landingImage") or soup.select_one("#imgBlkFront")

        title       = title_el.get_text(strip=True) if title_el else ""
        price       = _extract_price(price_el.get_text() if price_el else "")
        description = desc_el.get_text(strip=True)[:300] if desc_el else ""
        image_url   = img_el.get("src", "") if img_el else ""

    # ── OLX / Dubizzle ──
    elif "dubizzle" in url or "olx" in url:
        title_el = soup.select_one("h1") or soup.select_one("[data-testid='title']")
        price_el = soup.select_one("[data-testid='price']") or soup.select_one("span.price")
        desc_el  = soup.select_one("[data-testid='description']") or soup.select_one(".description")
        img_el   = soup.select_one("img[src*='cdn']") or soup.select_one("picture img")

        title       = title_el.get_text(strip=True) if title_el else ""
        price       = _extract_price(price_el.get_text() if price_el else "")
        description = desc_el.get_text(strip=True)[:300] if desc_el else ""
        image_url   = img_el.get("src", "") if img_el else ""

    # ── Generic fallback ──
    else:
        title_el = soup.find("h1")
        price_el = soup.find(string=re.compile(r"EGP|ج\.م|£E|\$", re.I))
        img_el   = soup.find("img")

        title       = title_el.get_text(strip=True) if title_el else ""
        price       = _extract_price(str(price_el) if price_el else "")
        description = (soup.find("meta", {"name": "description"}) or {}).get("content", "")[:300]
        image_url   = img_el.get("src", "") if img_el else ""

    # Fallbacks for missing title/price
    if not title:
        og = soup.find("meta", property="og:title")
        title = og["content"] if og else "Unknown Product"
    if not image_url:
        og_img = soup.find("meta", property="og:image")
        image_url = og_img["content"] if og_img else ""

    return {
        "title":       title,
        "price":       price,
        "description": description,
        "image_url":   image_url,
        "url":         url,
    }


def _extract_price(text: str) -> str:
    nums = re.findall(r"[\d,]+\.?\d*", text.replace(",", ""))
    for n in nums:
        try:
            v = float(n.replace(",", ""))
            if v > 100:
                return str(int(v))
        except:
            pass
    return ""


# ── Model runner (DUMMY — swap with real models) ───────────────────────────────
def run_models(product: dict) -> dict:
    """
    TODO: Replace each score with actual model inference.

    text_score  → XLM-RoBERTa / LSTM on title + description
    image_score → ResNet50 / EfficientNet on product image
    ml_score    → Random Forest / XGBoost on price + model features
    """
    time.sleep(1.5)  # simulate inference time

    # ── DUMMY SCORES — replace these ──
    text_score  = round(random.uniform(0.3, 0.9), 2)
    image_score = round(random.uniform(0.2, 0.85), 2)
    ml_score    = round(random.uniform(0.1, 0.95), 2)

    # Weighted fusion (text 40%, image 30%, ml 30%)
    final_score = round(0.4 * text_score + 0.3 * image_score + 0.3 * ml_score, 2)

    return {
        "text_score":  text_score,
        "image_score": image_score,
        "ml_score":    ml_score,
        "final_score": final_score,
    }


def get_reasons(scores: dict, product: dict) -> list[str]:
    """Generate human-readable risk reasons based on scores."""
    reasons = []
    if scores["text_score"] > 0.6:
        reasons.append("🔤 Suspicious language detected in title/description")
    if scores["image_score"] > 0.6:
        reasons.append("🖼️ Product image may not match the listed item")
    if scores["ml_score"] > 0.6:
        reasons.append("💰 Price is significantly below market average")
    if not product.get("description"):
        reasons.append("📋 No product description provided")
    if not product.get("price"):
        reasons.append("❓ Price is missing or unclear")
    if not reasons:
        reasons.append("✅ No major red flags detected")
    return reasons


# ── Gauge chart ───────────────────────────────────────────────────────────────
def render_gauge(score: float):
    """Render a simple SVG gauge for the risk score."""
    pct = score * 100
    if pct < 40:
        color = "#2ecc71"
        label = "LOW RISK"
        css_class = "safe"
    elif pct < 70:
        color = "#f39c12"
        label = "MEDIUM RISK"
        css_class = "medium"
    else:
        color = "#e74c3c"
        label = "HIGH RISK"
        css_class = "danger"

    # SVG arc gauge
    import math
    cx, cy, r = 100, 90, 70
    start_angle = 180
    end_angle   = 180 + (pct / 100) * 180
    x1 = cx + r * math.cos(math.radians(start_angle))
    y1 = cy + r * math.sin(math.radians(start_angle))
    x2 = cx + r * math.cos(math.radians(end_angle))
    y2 = cy + r * math.sin(math.radians(end_angle))
    large = 1 if (end_angle - start_angle) > 180 else 0

    svg = f"""
    <svg viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg">
      <!-- Background arc -->
      <path d="M 30 90 A 70 70 0 0 1 170 90" fill="none" stroke="#2a2a3e" stroke-width="16" stroke-linecap="round"/>
      <!-- Score arc -->
      <path d="M {x1:.1f} {y1:.1f} A 70 70 0 {large} 1 {x2:.1f} {y2:.1f}"
            fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"/>
      <!-- Score text -->
      <text x="100" y="85" text-anchor="middle" font-size="26" font-weight="bold" fill="{color}">{int(pct)}%</text>
      <text x="100" y="105" text-anchor="middle" font-size="11" fill="#888">{label}</text>
      <!-- Min/Max labels -->
      <text x="28"  y="108" text-anchor="middle" font-size="9" fill="#555">0%</text>
      <text x="172" y="108" text-anchor="middle" font-size="9" fill="#555">100%</text>
    </svg>
    """
    st.markdown(f'<div style="text-align:center">{svg}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="risk-box {css_class}"><b>{label}</b> — Scam probability: {int(pct)}%</div>',
                unsafe_allow_html=True)


# ── Recommendations ───────────────────────────────────────────────────────────
TRUSTED_SITES = {
    "🛒 Amazon Egypt": "https://www.amazon.eg/s?k={query}&i=electronics",
    "📦 Jumia Egypt":  "https://www.jumia.com.eg/catalog/?q={query}",
    "🌙 Noon Egypt":   "https://www.noon.com/egypt-en/search/?q={query}",
}

def render_recommendations(title: str):
    # Extract phone model keywords from title
    query = re.sub(r"[^\w\s]", "", title)
    query = " ".join(query.split()[:5])  # first 5 words
    encoded = quote_plus(query)

    st.markdown("### 🛡️ Find it from a trusted source")
    st.markdown("We found the same product on these verified stores:")

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
    st.markdown("Paste a product link below to check if it's safe to buy.")

    url = st.text_input(
        "Product URL",
        placeholder="https://www.dubizzle.com.eg/en/ad/iphone-15-pro-..."
    )

    analyze_btn = st.button("Analyze Product", type="primary", use_container_width=True)

    if analyze_btn and url:
        if not url.startswith("http"):
            st.error("Please enter a valid URL starting with http:// or https://")
            return

        # Step 1: Scrape
        with st.spinner("🔎 Fetching product details..."):
            product = scrape_product(url)

        if "error" in product:
            st.error(f"Could not fetch the product: {product['error']}")
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
            st.markdown(f"**{product['title'] or 'Unknown Title'}**")
            if product.get("price"):
                st.markdown(f"💵 **Price:** {product['price']} EGP")
            else:
                st.markdown("💵 **Price:** Not found")
            if product.get("description"):
                st.markdown(f"📋 {product['description'][:200]}...")

        # Step 2: Run models
        st.markdown("---")
        st.markdown("### 🤖 Running AI Analysis...")
        with st.spinner("Analyzing with text, image, and price models..."):
            scores = run_models(product)

        # Step 3: Show gauge
        st.markdown("### 📊 Risk Score")
        render_gauge(scores["final_score"])

        # Model breakdown
        with st.expander("See model breakdown"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Text Model",  f"{int(scores['text_score']*100)}%",  help="XLM-RoBERTa + LSTM")
            c2.metric("Image Model", f"{int(scores['image_score']*100)}%", help="ResNet50 + EfficientNet")
            c3.metric("Price Model", f"{int(scores['ml_score']*100)}%",    help="Random Forest + XGBoost")

        # Step 4: Reasons
        st.markdown("### ⚠️ Risk Factors")
        reasons = get_reasons(scores, product)
        for r in reasons:
            st.markdown(f'<div class="reason-box">{r}</div>', unsafe_allow_html=True)

        # Step 5: Recommendations
        st.markdown("---")
        render_recommendations(product["title"])

    elif analyze_btn and not url:
        st.warning("Please enter a product URL first.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#555; font-size:0.8rem'>"
        "Scam Detection System — HNU Deep Learning Project 2025"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
