"""
Scrape trusted mobile phone listings from:
  - Jumia Egypt      (target 100) — server-side rendered
  - Carrefour Egypt  (target 100) — replaces 2B (blocked)
  - Amazon Egypt     (target 100)

Each listing gets label=0 (trusted).
Saves images to Data/raw_data/images/trusted/
Appends rows to Data/raw_data/listings.csv
Writes Data/raw_data/prices.csv
"""

import os
import csv
import time
import random
import hashlib
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote_plus

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "raw_data")
IMG_DIR    = os.path.join(RAW_DIR, "images", "trusted")
CSV_PATH   = os.path.join(RAW_DIR, "listings.csv")
PRICES_PATH = os.path.join(RAW_DIR, "prices.csv")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

CSV_COLUMNS = [
    "id", "source_url", "source_site", "label", "title", "description",
    "price_listed", "phone_model", "image_paths", "seller_name", "seller_rating"
]

# Hardcoded baseline prices (EGP) — extended at runtime from scraped data
BASELINE_PRICES = {
    "iPhone 16 Pro Max":      {"min": 85000,  "max": 105000},
    "iPhone 16 Pro":          {"min": 70000,  "max": 88000},
    "iPhone 16":              {"min": 55000,  "max": 70000},
    "iPhone 15 Pro Max":      {"min": 72000,  "max": 90000},
    "iPhone 15 Pro":          {"min": 60000,  "max": 75000},
    "iPhone 15":              {"min": 48000,  "max": 62000},
    "iPhone 14 Pro Max":      {"min": 55000,  "max": 70000},
    "iPhone 14":              {"min": 38000,  "max": 50000},
    "iPhone 13":              {"min": 28000,  "max": 38000},
    "iPhone 12":              {"min": 20000,  "max": 28000},
    "Samsung Galaxy S25 Ultra": {"min": 80000, "max": 100000},
    "Samsung Galaxy S25":     {"min": 55000,  "max": 70000},
    "Samsung Galaxy S24 Ultra": {"min": 70000, "max": 88000},
    "Samsung Galaxy S24":     {"min": 48000,  "max": 62000},
    "Samsung Galaxy S23":     {"min": 35000,  "max": 48000},
    "Samsung Galaxy A55":     {"min": 18000,  "max": 25000},
    "Samsung Galaxy A35":     {"min": 13000,  "max": 18000},
    "Samsung Galaxy A15":     {"min": 7000,   "max": 10000},
    "Xiaomi Redmi Note 13 Pro": {"min": 15000, "max": 22000},
    "Xiaomi Redmi Note 13":   {"min": 10000,  "max": 15000},
    "Xiaomi 14 Pro":          {"min": 45000,  "max": 58000},
    "OPPO Reno 12 Pro":       {"min": 22000,  "max": 30000},
    "OPPO A60":               {"min": 8000,   "max": 12000},
    "Realme 12 Pro":          {"min": 14000,  "max": 20000},
    "Huawei Nova 12":         {"min": 18000,  "max": 25000},
    "Infinix Hot 40 Pro":     {"min": 6000,   "max": 9000},
    "Infinix Note 40 Pro":    {"min": 10000,  "max": 15000},
    "Tecno Spark 20 Pro":     {"min": 5000,   "max": 8000},
    "Tecno Camon 30":         {"min": 8000,   "max": 12000},
    "Nokia G42":              {"min": 6000,   "max": 9000},
}

price_data = {m: dict(v) for m, v in BASELINE_PRICES.items()}


# ── helpers ──────────────────────────────────────────────────────────────────

def get_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def delay(lo=2, hi=5):
    time.sleep(random.uniform(lo, hi))


def generate_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:8]


def extract_price(text):
    if not text:
        return ""
    cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
    nums = re.findall(r"\d+\.?\d*", cleaned)
    if nums:
        try:
            return str(int(float(nums[0])))
        except ValueError:
            pass
    return ""


def extract_phone_model(title):
    if not title:
        return ""
    patterns = [
        r"(iPhone\s*\d+\s*(?:Pro\s*Max|Pro|Plus|Mini)?(?:\s*\d+\s*(?:GB|TB))?)",
        r"(Samsung\s+Galaxy\s+[A-Z]\d+\s*(?:FE|Ultra|Plus|\+)?(?:\s*5G)?)",
        r"(Samsung\s+Galaxy\s+[A-Z]\d+[a-z]?\s*\d*)",
        r"(Xiaomi\s+(?:Redmi\s+)?(?:Note\s+)?\d+\s*(?:Pro|Ultra|Plus|\+)?(?:\s*5G)?)",
        r"(OPPO\s+(?:Reno|Find|A)\s*\d+\s*(?:Pro|Ultra|Plus|\+)?(?:\s*5G)?)",
        r"(Realme\s+\d+\s*(?:Pro|Ultra|Plus|\+)?(?:\s*5G)?)",
        r"(Huawei\s+(?:Nova|P|Mate)\s*\d+\s*(?:Pro|Lite|Ultra)?)",
        r"(OnePlus\s+\d+\s*(?:Pro|T|R)?)",
        r"(Google\s+Pixel\s+\d+\s*(?:Pro|a)?)",
        r"(Vivo\s+[A-Z]\d+\s*(?:Pro|Plus|\+)?(?:\s*5G)?)",
        r"(Nokia\s+[A-Z]?\d+)",
        r"(Tecno\s+(?:Spark|Camon|Pova|Phantom)\s*\d*\s*(?:Pro|Go|Plus)?)",
        r"(Infinix\s+(?:Hot|Note|Zero|Smart)\s*\d*\s*(?:Pro|Play|Plus|i)?)",
        r"(Honor\s+\d+\s*(?:Pro|Lite|X|Magic)?)",
        r"(Motorola\s+(?:Edge|Moto)\s*\w*\s*\d*)",
    ]
    for p in patterns:
        m = re.search(p, title, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return title[:50]


def update_prices(model, price):
    if not model or not price:
        return
    try:
        p = int(price)
    except (ValueError, TypeError):
        return
    if p < 1000:
        return
    if model not in price_data:
        price_data[model] = {"min": p, "max": p}
    else:
        price_data[model]["min"] = min(price_data[model]["min"], p)
        price_data[model]["max"] = max(price_data[model]["max"], p)


def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def get_existing_ids():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, encoding="utf-8") as f:
        return {r["id"] for r in csv.DictReader(f)}


def append_row(row):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)


def save_image(url, listing_id, index, session):
    try:
        r = session.get(url, timeout=15, stream=True)
        if r.status_code == 200:
            fname = f"{listing_id}_{index}.jpg"
            fpath = os.path.join(IMG_DIR, fname)
            with open(fpath, "wb") as f:
                for chunk in r.iter_content(1024):
                    f.write(chunk)
            return os.path.join("images", "trusted", fname)
    except Exception:
        pass
    return None


def make_row(listing_id, url, site, title, desc, price, model, img_paths, seller, rating):
    return {
        "id": listing_id,
        "source_url": url,
        "source_site": site,
        "label": 0,
        "title": title,
        "description": desc if desc else title,
        "price_listed": price,
        "phone_model": model,
        "image_paths": ",".join(img_paths),
        "seller_name": seller,
        "seller_rating": rating,
    }


# ── scrapers ──────────────────────────────────────────────────────────────────

def scrape_amazon_egypt(session, existing_ids, target=100):
    print(f"\n[Amazon Egypt] target={target}")
    rows = []
    seen = set(existing_ids)
    terms = ["iphone", "samsung galaxy", "xiaomi redmi", "oppo", "infinix", "tecno", "realme", "huawei nova"]

    for term in terms:
        if len(rows) >= target:
            break
        for page in range(1, 8):
            if len(rows) >= target:
                break
            url = f"https://www.amazon.eg/s?k={quote_plus(term)}&i=electronics&page={page}"
            try:
                resp = session.get(url, timeout=25)
                if resp.status_code != 200:
                    print(f"  [Amazon] {term} p{page} → {resp.status_code}")
                    delay(3, 7)
                    continue
                resp.encoding = 'utf-8'
                soup = BeautifulSoup(resp.text, "html.parser")
                items = soup.select('[data-component-type="s-search-result"]') or \
                        soup.select(".s-result-item[data-asin]")

                for item in items:
                    if len(rows) >= target:
                        break
                    asin = item.get("data-asin", "")
                    if not asin:
                        continue
                    title_el = item.select_one("h2 a span") or item.select_one("h2 span")
                    if not title_el:
                        continue
                    title = title_el.get_text(strip=True)
                    if not title:
                        continue
                    link_el = item.select_one("h2 a")
                    product_url = urljoin("https://www.amazon.eg", link_el["href"]) if link_el else url
                    lid = generate_id(product_url)
                    if lid in seen:
                        continue
                    seen.add(lid)

                    price_el = item.select_one(".a-price-whole") or item.select_one(".a-offscreen")
                    price = extract_price(price_el.get_text() if price_el else "")
                    rating_el = item.select_one(".a-icon-alt")
                    rating = ""
                    if rating_el:
                        m = re.search(r"([\d.]+)", rating_el.get_text())
                        if m:
                            rating = m.group(1)
                    img_el = item.select_one("img.s-image")
                    img_paths = []
                    if img_el:
                        saved = save_image(img_el["src"], lid, 0, session)
                        if saved:
                            img_paths.append(saved)

                    model = extract_phone_model(title)
                    row = make_row(lid, product_url, "amazon_egypt", title, title, price, model, img_paths, "Amazon Egypt", rating)
                    append_row(row)
                    update_prices(model, price)
                    rows.append(row)
                    print(f"  [Amazon] {len(rows)}/{target}: {title[:55]}")

                delay()
            except Exception as e:
                print(f"  [Amazon] error: {e}")
                delay(3, 7)

    print(f"[Amazon Egypt] done — {len(rows)} rows")
    return rows


def scrape_extra_egypt(session, existing_ids, target=100):
    """
    eXtra Egypt — electronics retailer, server-side rendered.
    Mobile phones: https://www.extra.com/en-eg/mobile-phones/c/CM010000
    """
    print(f"\n[eXtra Egypt] target={target}")
    rows = []
    seen = set(existing_ids)

    search_terms = ["iphone", "samsung", "xiaomi", "oppo", "infinix",
                    "realme", "tecno", "huawei", "nokia", "vivo"]

    for term in search_terms:
        if len(rows) >= target:
            break
        for page in range(1, 6):
            if len(rows) >= target:
                break
            url = f"https://www.extra.com/en-eg/search/?q={quote_plus(term)}&page={page}"
            try:
                resp = session.get(url, timeout=25)
                if resp.status_code != 200:
                    print(f"  [eXtra] {term} p{page} → {resp.status_code}")
                    delay(3, 6)
                    continue
                resp.encoding = 'utf-8'
                soup = BeautifulSoup(resp.text, "html.parser")

                # eXtra product cards
                items = (soup.select("li.product-item") or
                         soup.select(".product-tile") or
                         soup.select("[class*='product-card']") or
                         soup.select("div.product") or
                         soup.select("article.product"))

                if not items:
                    # fallback: grab all product links
                    product_links = [
                        a for a in soup.select("a[href*='/en-eg/']")
                        if a.select_one("img") and len(a.get_text(strip=True)) > 5
                    ]
                    print(f"  [eXtra] {term} p{page} → {len(product_links)} links (fallback)")
                    for link in product_links:
                        if len(rows) >= target:
                            break
                        href = link.get("href", "")
                        if not href or "/search" in href or "/category" in href:
                            continue
                        product_url = urljoin("https://www.extra.com", href)
                        lid = generate_id(product_url)
                        if lid in seen:
                            continue
                        seen.add(lid)

                        title = link.get_text(strip=True)
                        if not title or len(title) < 5:
                            img = link.select_one("img")
                            title = img.get("alt", "") if img else ""
                        if not title or len(title) < 5:
                            continue

                        img_el = link.select_one("img")
                        img_url = img_el.get("src", "") or img_el.get("data-src", "") if img_el else ""
                        img_paths = []
                        if img_url and img_url.startswith("http"):
                            saved = save_image(img_url, lid, 0, session)
                            if saved:
                                img_paths.append(saved)

                        model = extract_phone_model(title)
                        row = make_row(lid, product_url, "extra_egypt", title, title, "", model, img_paths, "eXtra Egypt", "")
                        append_row(row)
                        rows.append(row)
                        print(f"  [eXtra] {len(rows)}/{target}: {title[:55]}")
                    delay()
                    continue

                print(f"  [eXtra] {term} p{page} → {len(items)} cards")

                for item in items:
                    if len(rows) >= target:
                        break

                    link_el = item.select_one("a")
                    if not link_el:
                        continue
                    href = link_el.get("href", "")
                    if not href:
                        continue
                    product_url = urljoin("https://www.extra.com", href)
                    lid = generate_id(product_url)
                    if lid in seen:
                        continue
                    seen.add(lid)

                    # Title
                    title_el = (item.select_one("h2") or item.select_one("h3") or
                                item.select_one("[class*='title']") or item.select_one("[class*='name']"))
                    title = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
                    if not title or len(title) < 5:
                        img = item.select_one("img")
                        title = img.get("alt", "") if img else ""
                    if not title or len(title) < 5:
                        continue

                    # Price
                    price_el = (item.select_one("[class*='price']") or
                                item.select_one("[class*='Price']"))
                    price = extract_price(price_el.get_text() if price_el else "")

                    # Image
                    img_el = item.select_one("img")
                    img_url = ""
                    if img_el:
                        img_url = img_el.get("src", "") or img_el.get("data-src", "")
                    img_paths = []
                    if img_url and img_url.startswith("http"):
                        saved = save_image(img_url, lid, 0, session)
                        if saved:
                            img_paths.append(saved)

                    model = extract_phone_model(title)
                    row = make_row(lid, product_url, "extra_egypt", title, title, price, model, img_paths, "eXtra Egypt", "")
                    append_row(row)
                    update_prices(model, price)
                    rows.append(row)
                    print(f"  [eXtra] {len(rows)}/{target}: {title[:55]}")

                delay()
            except Exception as e:
                print(f"  [eXtra] error: {e}")
                delay(3, 6)

    print(f"[eXtra Egypt] done — {len(rows)} rows")
    return rows


def scrape_jumia_egypt(session, existing_ids, target=100):
    """
    Jumia Egypt — server-side rendered, much easier than Noon.
    Category page: https://www.jumia.com.eg/mobile-phones/
    """
    print(f"\n[Jumia Egypt] target={target}")
    rows = []
    seen = set(existing_ids)

    for page in range(1, 20):
        if len(rows) >= target:
            break
        url = f"https://www.jumia.com.eg/mobile-phones/?page={page}#catalog-listing"
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                print(f"  [Jumia] page {page} → {resp.status_code}")
                delay()
                continue
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, "html.parser")

            # Jumia product cards
            items = soup.select("article.prd") or soup.select(".sku-item") or soup.select("article")
            if not items:
                print(f"  [Jumia] page {page} → no items, stopping")
                break

            print(f"  [Jumia] page {page} → {len(items)} items")

            for item in items:
                if len(rows) >= target:
                    break

                link_el = item.select_one("a.core") or item.select_one("a[href*='/mlp']") or item.select_one("a")
                if not link_el:
                    continue
                href = link_el.get("href", "")
                if not href:
                    continue
                product_url = urljoin("https://www.jumia.com.eg", href)
                lid = generate_id(product_url)
                if lid in seen:
                    continue
                seen.add(lid)

                # Title
                title_el = item.select_one(".name") or item.select_one("h3") or item.select_one(".title")
                title = title_el.get_text(strip=True) if title_el else link_el.get_text(strip=True)
                if not title or len(title) < 5:
                    continue

                # Price
                price_el = item.select_one(".prc") or item.select_one("[class*='price']")
                price = extract_price(price_el.get_text() if price_el else "")

                # Rating
                rating_el = item.select_one(".stars._s") or item.select_one("[class*='stars']")
                rating = ""
                if rating_el:
                    m = re.search(r"([\d.]+)", rating_el.get("style", "") + rating_el.get_text())
                    if m:
                        rating = m.group(1)

                # Image
                img_el = item.select_one("img")
                img_url = ""
                if img_el:
                    img_url = img_el.get("data-src", "") or img_el.get("src", "")
                img_paths = []
                if img_url and img_url.startswith("http"):
                    saved = save_image(img_url, lid, 0, session)
                    if saved:
                        img_paths.append(saved)

                model = extract_phone_model(title)
                row = make_row(lid, product_url, "jumia_egypt", title, title, price, model, img_paths, "Jumia Egypt", rating)
                append_row(row)
                update_prices(model, price)
                rows.append(row)
                print(f"  [Jumia] {len(rows)}/{target}: {title[:55]}")

            delay()
        except Exception as e:
            print(f"  [Jumia] error: {e}")
            delay()

    print(f"[Jumia Egypt] done — {len(rows)} rows")
    return rows


# ── prices ────────────────────────────────────────────────────────────────────

def save_prices():
    # Merge with existing prices.csv if present
    existing = {}
    if os.path.exists(PRICES_PATH):
        with open(PRICES_PATH, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                try:
                    existing[r["phone_model"]] = {
                        "min": int(r["price_min"]),
                        "max": int(r["price_max"]),
                    }
                except (ValueError, KeyError):
                    pass
    for model, vals in price_data.items():
        if model in existing:
            existing[model]["min"] = min(existing[model]["min"], vals["min"])
            existing[model]["max"] = max(existing[model]["max"], vals["max"])
        else:
            existing[model] = vals

    with open(PRICES_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["phone_model", "price_min", "price_max"])
        w.writeheader()
        for model in sorted(existing):
            w.writerow({"phone_model": model, "price_min": existing[model]["min"], "price_max": existing[model]["max"]})
    print(f"\n[Prices] Saved {len(existing)} models to prices.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    init_csv()
    session = get_session()
    existing_ids = get_existing_ids()
    print(f"Existing rows in CSV: {len(existing_ids)}")

    all_rows = []
    all_rows.extend(scrape_jumia_egypt(session, existing_ids, target=100))
    existing_ids = get_existing_ids()
    all_rows.extend(scrape_extra_egypt(session, existing_ids, target=100))
    existing_ids = get_existing_ids()
    all_rows.extend(scrape_amazon_egypt(session, existing_ids, target=100))

    save_prices()

    print(f"\n{'='*55}")
    print(f"TRUSTED SCRAPING COMPLETE")
    print(f"New rows this run : {len(all_rows)}")
    print(f"CSV               : {CSV_PATH}")
    print(f"Images            : {IMG_DIR}")
    print(f"Prices            : {PRICES_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
