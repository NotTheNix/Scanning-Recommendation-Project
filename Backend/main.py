"""
Backend API — Scam Detection & Recommendation System
======================================================
Framework : FastAPI
Run       : uvicorn main:app --reload  (from Backend/ folder)
Docs      : http://127.0.0.1:8000/docs

Endpoints:
  POST /scrape       — scrape a listing URL → structured data
  POST /scan         — run fusion model on listing data → scam score
  GET  /recommend    — return safe alternative listings
  GET  /health       — check if API is alive

Install:
  pip install fastapi uvicorn requests beautifulsoup4
"""

import os, sys, re, warnings
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

warnings.filterwarnings("ignore")

# ── Add Models/ to path so we can import fusion models ────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
sys.path.insert(0, os.path.join(MODELS_DIR, "fusion"))
sys.path.insert(0, os.path.join(MODELS_DIR, "text_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "image_models"))
sys.path.insert(0, os.path.join(MODELS_DIR, "ml_models"))

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Scam Detection API",
    description = "Mobile phone listing scam detector powered by multimodal fusion.",
    version     = "1.0.0",
)

# Allow all origins (for Streamlit UI to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────
class ScrapeRequest(BaseModel):
    url: str


class ScrapeResponse(BaseModel):
    url:           str
    title:         Optional[str] = None
    description:   Optional[str] = None
    price:         Optional[float] = None
    phone_model:   Optional[str] = None
    image_url:     Optional[str] = None
    seller_rating: Optional[float] = None
    source:        Optional[str] = None
    success:       bool = True
    error:         Optional[str] = None


class ScanRequest(BaseModel):
    title:         str
    description:   Optional[str] = ""
    price:         Optional[float] = None
    phone_model:   Optional[str] = "unknown"
    image_path:    Optional[str] = ""
    seller_rating: Optional[float] = None
    fusion:        Optional[str] = "A"   # "A" = LSTM+ResNet50+XGB, "B" = TFIDF+EfficientNet+RF


class ScoreBreakdown(BaseModel):
    model_name: str
    score:      float
    weight:     float
    contribution: float


class ScanResponse(BaseModel):
    final_score:  float
    verdict:      str
    risk_level:   str
    fusion:       str
    breakdown:    list[ScoreBreakdown]


class RecommendResponse(BaseModel):
    recommendations: list[dict]


# ── Scraper helper ────────────────────────────────────────────────────────────
def scrape_url(url: str) -> dict:
    """
    Scrapes a listing URL and returns structured data.
    Supports: Jumia Egypt, Amazon Egypt, OLX/Dubizzle, generic fallback.
    """
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        return {"success": False, "error": str(e), "url": url}

    result = {"url": url, "success": True}

    # ── Jumia Egypt ──
    if "jumia.com.eg" in url:
        result["source"] = "Jumia Egypt"
        try:
            result["title"] = soup.find("h1", class_="-fs20").get_text(strip=True)
        except Exception:
            pass
        try:
            result["price"] = float(
                re.sub(r"[^\d.]", "", soup.find("span", class_="-b -ltr -tal -fs24").get_text())
            )
        except Exception:
            pass
        try:
            result["image_url"] = soup.find("img", class_="img-a _m img -fw -fh")["src"]
        except Exception:
            pass

    # ── Amazon Egypt ──
    elif "amazon.eg" in url or "amazon.com" in url:
        result["source"] = "Amazon Egypt"
        try:
            result["title"] = soup.find("span", id="productTitle").get_text(strip=True)
        except Exception:
            pass
        try:
            price_whole = soup.find("span", class_="a-price-whole")
            price_frac  = soup.find("span", class_="a-price-fraction")
            if price_whole:
                price_str = price_whole.get_text(strip=True).replace(",", "")
                if price_frac:
                    price_str += "." + price_frac.get_text(strip=True)
                result["price"] = float(price_str)
        except Exception:
            pass
        try:
            result["image_url"] = soup.find("img", id="landingImage")["src"]
        except Exception:
            pass

    # ── OLX / Dubizzle ──
    elif "olx" in url or "dubizzle" in url:
        result["source"] = "OLX"
        try:
            result["title"] = soup.find("h1").get_text(strip=True)
        except Exception:
            pass
        try:
            price_tag = soup.find("span", attrs={"data-aut-id": "itemPrice"})
            result["price"] = float(re.sub(r"[^\d.]", "", price_tag.get_text()))
        except Exception:
            pass
        try:
            result["description"] = soup.find(
                "span", attrs={"data-aut-id": "itemDescriptionContent"}
            ).get_text(strip=True)
        except Exception:
            pass

    # ── Generic fallback ──
    else:
        result["source"] = "Unknown"
        try:
            result["title"] = soup.find("h1").get_text(strip=True)
        except Exception:
            pass
        try:
            og_image = soup.find("meta", property="og:image")
            if og_image:
                result["image_url"] = og_image["content"]
        except Exception:
            pass
        try:
            og_desc = soup.find("meta", property="og:description")
            if og_desc:
                result["description"] = og_desc["content"]
        except Exception:
            pass

    return result


# ── Recommendation helper ─────────────────────────────────────────────────────
RECOMMENDATIONS = [
    {
        "platform": "Jumia Egypt",
        "url":      "https://www.jumia.com.eg/mobile-phones/",
        "logo":     "https://www.jumia.com.eg/favicon.ico",
        "note":     "Egypt's largest verified e-commerce platform",
    },
    {
        "platform": "Amazon Egypt",
        "url":      "https://www.amazon.eg/s?k=smartphones",
        "logo":     "https://www.amazon.eg/favicon.ico",
        "note":     "Trusted global marketplace with buyer protection",
    },
    {
        "platform": "Noon Egypt",
        "url":      "https://www.noon.com/egypt-en/mobiles-and-tablets/mobiles-c-17/?q=smartphone",
        "logo":     "https://www.noon.com/favicon.ico",
        "note":     "Regional platform with official warranties",
    },
]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check if the API is running."""
    return {"status": "ok", "message": "Scam Detection API is running"}


@app.post("/scrape", response_model=ScrapeResponse)
def scrape(req: ScrapeRequest):
    """
    Scrape a product listing URL and return structured data.
    Supports Jumia EG, Amazon EG, OLX, and generic pages.
    """
    data = scrape_url(req.url)
    if not data.get("success"):
        raise HTTPException(status_code=422, detail=data.get("error", "Scraping failed"))
    return ScrapeResponse(**data)


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    """
    Run the fusion model on a listing and return scam probability + breakdown.
    Set fusion='A' for LSTM+ResNet50+XGBoost, fusion='B' for TFIDF+EfficientNet+RF.
    """
    fusion_choice = (req.fusion or "A").upper()

    try:
        if fusion_choice == "B":
            import fusion_model_b as fm
        else:
            import fusion_model as fm

        result = fm.predict(
            title         = req.title,
            description   = req.description or "",
            price         = req.price,
            phone_model   = req.phone_model or "unknown",
            image_path    = req.image_path or "",
            seller_rating = req.seller_rating,
            verbose       = False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    breakdown = [
        ScoreBreakdown(
            model_name   = model_name,
            score        = round(score, 4),
            weight       = result["weights"][model_name],
            contribution = round(result["weights"][model_name] * score, 4),
        )
        for model_name, score in result["scores"].items()
    ]

    return ScanResponse(
        final_score = result["final_score"],
        verdict     = result["verdict"],
        risk_level  = result["risk_level"],
        fusion      = fusion_choice,
        breakdown   = breakdown,
    )


@app.get("/recommend", response_model=RecommendResponse)
def recommend():
    """
    Return a list of trusted platforms to buy phones safely.
    """
    return RecommendResponse(recommendations=RECOMMENDATIONS)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
