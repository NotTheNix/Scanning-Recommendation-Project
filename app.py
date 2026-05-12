"""
Scam Detection & Recommendation System — Streamlit UI
======================================================
No backend required. Models are called directly.

Run:
  streamlit run app.py
"""

import re
import os
import sys
import math
import tempfile
import warnings
import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

warnings.filterwarnings("ignore")

# ── Add model paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
sys.path.insert(0, os.path.join(MODELS_DIR, "fusion"))
sys.path.insert(0, os.path.join(MODELS_DIR, "text_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "image_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "ml_models"))

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
</style>
""", unsafe_allow_html=True)


# ── Scraper ───────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

def _extract_price(text: str) -> float | None:
    """Extract first numeric value > 100 from a string as float."""
    cleaned = text.replace(",", "").replace("\xa0", "")
    nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
    for n in nums:
        try:
            v = float(n)
            if v > 100:
                return v
        except Exception:
            pass
    return None


def scrape_product(url: str) -> dict:
    """
    Scrape title, price, description, image from a product URL.
    Supports: Jumia EG, Amazon EG, OLX/Dubizzle, Noon, generic fallback.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return {"error": str(e)}

    title = price = description = image_url = source = ""

    # ── Jumia Egypt ──
    if "jumia.com" in url:
        source = "Jumia Egypt"
        # Title
        for sel in ["h1.-fs20", "h1.-fs20.-pts.-pbxs", "h1"]:
            el = soup.select_one(sel)
            if el:
                title = el.get_text(strip=True)
                break
        # Price — try multiple selectors
        for sel in [".-b.-ltr.-tal.-fs24", "span.-b.-ltr", ".-fs24"]:
            el = soup.select_one(sel)
            if el:
                price = _extract_price(el.get_text())
                if price:
                    break
        # Description
        el = soup.select_one(".-mhm.-pvl.-mod.article") or soup.select_one(".markup")
        description = el.get_text(strip=True)[:300] if el else ""
        # Image — try data-src first (lazy loaded), then src
        for sel in ["img.-fw.-fh", "img.img-a", "img[data-src]"]:
            el = soup.select_one(sel)
            if el:
                image_url = el.get("data-src") or el.get("src", "")
                if image_url and image_url.startswith("http"):
                    break

    # ── Amazon Egypt ──
    elif "amazon.eg" in url or "amazon.com" in url:
        source = "Amazon Egypt"
        el = soup.select_one("#productTitle")
        title = el.get_text(strip=True) if el else ""
        # Price — whole + fraction
        whole = soup.select_one(".a-price-whole")
        frac  = soup.select_one(".a-price-fraction")
        if whole:
            price_str = whole.get_text(strip=True).replace(",", "").replace(".", "")
            if frac:
                price_str += "." + frac.get_text(strip=True)
            price = _extract_price(price_str)
        if not price:
            # Fallback: look for any .a-offscreen price
            el = soup.select_one(".a-offscreen")
            if el:
                price = _extract_price(el.get_text())
        # Description
        el = soup.select_one("#feature-bullets")
        description = el.get_text(strip=True)[:300] if el else ""
        # Image
        el = soup.select_one("#landingImage") or soup.select_one("#imgBlkFront")
        if el:
            image_url = el.get("data-old-hires") or el.get("src", "")

    # ── OLX / Dubizzle ──
    elif "olx" in url or "dubizzle" in url:
        source = "OLX"
        # Title
        el = soup.select_one("h1") or soup.select_one("[data-testid='title']")
        title = el.get_text(strip=True) if el else ""
        # Price — try multiple selectors
        for sel in ["[data-testid='price']", "span.price", "strong.price", "[class*='price']"]:
            el = soup.select_one(sel)
            if el:
                price = _extract_price(el.get_text())
                if price:
                    break
        # Description
        el = soup.select_one("[data-testid='description']") or soup.select_one(".description")
        description = el.get_text(strip=True)[:300] if el else ""
        # Image — OLX uses picture/source tags with high-res images
        el = soup.select_one("picture source[srcset]")
        if el:
            srcset = el.get("srcset", "")
            # Take the last (largest) URL from srcset
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            image_url = parts[-1] if parts else ""
        if not image_url:
            el = soup.select_one("img[src*='cdn']") or soup.select_one("picture img")
            image_url = el.get("src", "") if el else ""

    # ── Noon ──
    elif "noon.com" in url:
        source = "Noon"
        el = soup.select_one("h1") or soup.select_one("[class*='productTitle']")
        title = el.get_text(strip=True) if el else ""
        el = soup.select_one("[class*='price']")
        price = _extract_price(el.get_text()) if el else None
        el = soup.select_one("img[class*='product']") or soup.select_one("img[src*='cdn']")
        image_url = el.get("src", "") if el else ""

    # ── Generic fallback ──
    else:
        source = "Unknown"
        el = soup.find("h1")
        title = el.get_text(strip=True) if el else ""
        # og:image is most reliable for generic pages
        og_img = soup.find("meta", property="og:image")
        image_url = og_img["content"] if og_img else ""
        og_desc = soup.find("meta", property="og:description")
        description = og_desc["content"][:300] if og_desc else ""
        # Try to find price anywhere on the page
        price_el = soup.find(string=re.compile(r"EGP|ج\.م|£E", re.I))
        price = _extract_price(str(price_el)) if price_el else None

    # ── Fallbacks ──
    if not title:
        og = soup.find("meta", property="og:title")
        title = og["content"] if og else ""
    if not image_url:
        og_img = soup.find("meta", property="og:image")
        image_url = og_img["content"] if og_img else ""

    # Make sure image_url is absolute
    if image_url and not image_url.startswith("http"):
        image_url = ""

    return {
        "title":       title,
        "price":       price,
        "description": description,
        "image_url":   image_url,
        "source":      source,
        "url":         url,
    }


# ── Phone model extractor ─────────────────────────────────────────────────────
PHONE_PATTERNS = [
    r"iphone\s*\d+\s*(?:pro\s*max|pro|plus|mini)?",
    r"samsung\s*galaxy\s*[a-z]\d+\s*(?:ultra|plus|fe)?",
    r"samsung\s*galaxy\s*s\d+\s*(?:ultra|plus|fe)?",
    r"xiaomi\s*\d+\s*(?:pro|ultra)?",
    r"redmi\s*note\s*\d+\s*(?:pro)?",
    r"oppo\s*[a-z]\d+\s*(?:pro)?",
    r"huawei\s*[a-z0-9\s]+",
    r"realme\s*\d+\s*(?:pro)?",
    r"vivo\s*[a-z0-9]+",
    r"oneplus\s*\d+\s*(?:pro|t)?",
]

def extract_phone_model(title: str) -> str:
    if not title:
        return "unknown"
    for pattern in PHONE_PATTERNS:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            return match.group(0).strip().title()
    return "unknown"


# ── Image downloader ──────────────────────────────────────────────────────────
def download_image(image_url: str) -> str:
    """Download image to temp file. Returns local path or ''."""
    if not image_url or not image_url.startswith("http"):
        return ""
    try:
        resp = requests.get(image_url, headers=HEADERS, timeout=10, stream=True)
        resp.raise_for_status()
        suffix = ".jpg"
        if "png" in image_url.lower():
            suffix = ".png"
        elif "webp" in image_url.lower():
            suffix = ".webp"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in resp.iter_content(chunk_size=8192):
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception:
        return ""


# ── Run fusion model directly ─────────────────────────────────────────────────
@st.cache_resource
def load_fusion_a():
    import fusion_model as fm
    return fm

@st.cache_resource
def load_fusion_b():
    import fusion_model_b as fm
    return fm

def run_fusion(product: dict, fusion_choice: str, local_image_path: str) -> dict:
    """Call the fusion model directly and return the result dict."""
    raw_price = product.get("price")
    try:
        price = float(str(raw_price).replace(",", "")) if raw_price else None
    except Exception:
        price = None

    phone_model   = extract_phone_model(product.get("title", ""))
    fm            = load_fusion_a() if fusion_choice == "A" else load_fusion_b()

    return fm.predict(
        title         = str(product.get("title") or ""),
        description   = str(product.get("description") or ""),
        price         = price,
        phone_model   = phone_model,
        image_path    = local_image_path,
        seller_rating = None,
        verbose       = False,
    )


# ── Gauge chart ───────────────────────────────────────────────────────────────
def render_gauge(score: float):
    pct = score * 100
    if pct < 40:
        color, label, css_class = "#2ecc71", "LOW RISK", "safe"
    elif pct < 65:
        color, label, css_class = "#f39c12", "MEDIUM RISK", "medium"
    else:
        color, label, css_class = "#e74c3c", "HIGH RISK", "danger"

    cx, cy, r_arc = 100, 90, 70
    end_angle = 180 + (pct / 100) * 180
    x1 = cx + r_arc * math.cos(math.radians(180))
    y1 = cy + r_arc * math.sin(math.radians(180))
    x2 = cx + r_arc * math.cos(math.radians(end_angle))
    y2 = cy + r_arc * math.sin(math.radians(end_angle))
    large = 1 if end_angle - 180 > 180 else 0

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
    reasons = []
    scores  = scan_result.get("scores", {})

    text_score = scores.get("lstm", scores.get("tfidf", 0))
    if text_score > 0.6:
        reasons.append(("🔤 Suspicious language detected in title or description", True))

    img_score = scores.get("resnet50", scores.get("efficientnet", 0))
    if img_score > 0.6:
        reasons.append(("🖼️ Product image shows signs associated with scam listings", True))

    ml_score = scores.get("xgboost", scores.get("random_forest", 0))
    if ml_score > 0.6:
        reasons.append(("💰 Price or seller details match known scam patterns", True))

    if not product.get("description"):
        reasons.append(("📋 No product description provided — common in scam listings", True))
    if not product.get("price"):
        reasons.append(("❓ Price is missing or could not be detected", True))

    if not reasons:
        reasons.append(("✅ No major red flags detected", False))

    return reasons


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

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Settings")
        fusion_choice = st.radio(
            "Fusion Model",
            options=["A", "B"],
            format_func=lambda x: (
                "Sequence A — LSTM + ResNet50 + XGBoost"
                if x == "A"
                else "Sequence B — TF-IDF + EfficientNet + RF"
            ),
            index=0,
        )
        st.markdown("---")
        st.markdown("**Sequence A** (recommended)")
        st.markdown("• LSTM 40% · ResNet50 35% · XGBoost 25%")
        st.markdown("**Sequence B**")
        st.markdown("• TF-IDF 40% · EfficientNet 35% · RF 25%")

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
        with st.spinner("🔎 Fetching product details..."):
            product = scrape_product(url)

        if "error" in product:
            st.error(f"Could not fetch the page: {product['error']}")
            return

        title = product.get("title", "").strip()
        if not title:
            st.error(
                "Could not extract product details from this page. "
                "The page may require login, use JavaScript, or block scrapers. "
                "Try a direct product link from Jumia, Amazon EG, or OLX."
            )
            return

        # ── Step 2: Download image ──
        image_url        = product.get("image_url", "")
        local_image_path = ""
        if image_url:
            with st.spinner("🖼️ Downloading product image..."):
                local_image_path = download_image(image_url)

        # ── Product card ──
        st.markdown("---")
        st.markdown("### 📱 Product Details")
        col1, col2 = st.columns([1, 2])
        with col1:
            if image_url:
                st.image(image_url, use_container_width=True)
            else:
                st.markdown("🖼️ *No image found*")
        with col2:
            st.markdown(f"**{title}**")
            if product.get("price"):
                st.markdown(f"💵 **Price:** {int(product['price']):,} EGP")
            else:
                st.markdown("💵 **Price:** Not found")
            if product.get("description"):
                st.markdown(f"📋 {product['description'][:200]}...")
            if product.get("source"):
                st.markdown(f"🌐 **Source:** {product['source']}")
            phone_model = extract_phone_model(title)
            if phone_model != "unknown":
                st.markdown(f"📱 **Detected Model:** {phone_model}")

        # ── Step 3: Run models ──
        st.markdown("---")
        st.markdown(f"### 🤖 Running AI Analysis (Fusion {fusion_choice})...")
        with st.spinner("Analyzing with text, image, and tabular models..."):
            try:
                scan_result = run_fusion(product, fusion_choice, local_image_path)
            except Exception as e:
                st.error(f"Model analysis failed: {e}")
                return
            finally:
                if local_image_path and os.path.exists(local_image_path):
                    try:
                        os.remove(local_image_path)
                    except Exception:
                        pass

        # ── Step 4: Gauge ──
        st.markdown("### 📊 Risk Score")
        render_gauge(scan_result["final_score"])

        verdict = scan_result["verdict"]
        verdict_color = {"Trusted": "#2ecc71", "Suspicious": "#f39c12", "Scam": "#e74c3c"}.get(verdict, "#888")
        st.markdown(
            f'<div style="text-align:center; font-size:1.3rem; font-weight:bold;'
            f' color:{verdict_color}; margin:8px 0">Verdict: {verdict}</div>',
            unsafe_allow_html=True,
        )

        # ── Step 5: Risk factors (only if score > 59%) ──
        if scan_result["final_score"] > 0.59:
            st.markdown("### ⚠️ Risk Factors")
            for reason_text, is_danger in get_reasons(scan_result, product):
                css = "reason-box" if is_danger else "ok-box"
                st.markdown(f'<div class="{css}">{reason_text}</div>', unsafe_allow_html=True)

        # ── Step 6: Recommendations ──
        st.markdown("---")
        render_recommendations(title)

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
