"""
Scrape scammy mobile phone listings from:
  - OLX Egypt / dubizzle.com.eg  (target 150)
  - OpenSooq                     (target 150) — replaces AliExpress (blocked)

Each listing gets label=1 (scam).
Saves images to Data/raw_data/images/scammy/
Appends rows to Data/raw_data/listings.csv
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

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "raw_data")
IMG_DIR     = os.path.join(RAW_DIR, "images", "scammy")
CSV_PATH    = os.path.join(RAW_DIR, "listings.csv")
PRICES_PATH = os.path.join(RAW_DIR, "prices.csv")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

OPENSOOQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://eg.opensooq.com/",
}

CSV_COLUMNS = [
    "id", "source_url", "source_site", "label", "title", "description",
    "price_listed", "phone_model", "image_paths", "seller_name", "seller_rating"
]

price_data = {}


# ── helpers ──────────────────────────────────────────────────────────────────

def get_session(extra_headers=None):
    s = requests.Session()
    s.headers.update(HEADERS)
    if extra_headers:
        s.headers.update(extra_headers)
    return s


def delay(lo=2, hi=5):
    time.sleep(random.uniform(lo, hi))


def generate_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:8]


def extract_price(text):
    if not text:
        return ""
    cleaned = text.replace(",", "").replace("EGP", "").replace("AED", "").replace("$", "").strip()
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
        r"(Mi\s+\d+\s*(?:Pro|Ultra|T|Lite)?(?:\s*5G)?)",
        r"(Honor\s+\d+\s*(?:Pro|Lite|X|Magic)?)",
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
    if p < 500:
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
            return os.path.join("images", "scammy", fname)
    except Exception:
        pass
    return None


def make_row(listing_id, url, site, title, desc, price, model, img_paths, seller, rating):
    return {
        "id": listing_id,
        "source_url": url,
        "source_site": site,
        "label": 1,
        "title": title,
        "description": desc if desc else title,
        "price_listed": price,
        "phone_model": model,
        "image_paths": ",".join(img_paths),
        "seller_name": seller,
        "seller_rating": rating,
    }


# ── scrapers ──────────────────────────────────────────────────────────────────

def scrape_olx_egypt(session, existing_ids, target=150):
    """
    OLX Egypt = dubizzle.com.eg — server-side rendered listing cards.
    Selector: li[aria-label='Listing']
    """
    print(f"\n[OLX Egypt] target={target}")
    rows = []
    seen = set(existing_ids)
    base = "https://www.dubizzle.com.eg/en/mobile-phones-tablets-accessories-numbers/mobile-phones/"

    for page in range(1, 25):
        if len(rows) >= target:
            break
        url = f"{base}?page={page}"
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                print(f"  [OLX] page {page} → {resp.status_code}")
                delay(3, 7)
                continue
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, "html.parser")

            items = soup.select("li[aria-label='Listing']")
            if not items:
                # fallback selectors
                items = soup.select("[data-aut-id='itemBox']") or soup.select("article")
            if not items:
                print(f"  [OLX] page {page} → no items, stopping")
                break

            print(f"  [OLX] page {page} → {len(items)} items")

            for item in items:
                if len(rows) >= target:
                    break

                link_el = item.select_one("a")
                if not link_el:
                    continue
                href = link_el.get("href", "")
                product_url = urljoin("https://www.dubizzle.com.eg", href)
                lid = generate_id(product_url)
                if lid in seen:
                    continue
                seen.add(lid)

                # Title — pick first div text that looks like a product name
                title = ""
                for el in item.select("h2, h3, [data-aut-id='itemTitle']"):
                    t = el.get_text(strip=True)
                    if t and len(t) > 3:
                        title = t
                        break
                if not title:
                    for div in item.select("div"):
                        t = div.get_text(strip=True)
                        if t and 5 < len(t) < 200 and not t.startswith("EGP") and "ago" not in t.lower():
                            if not any(x in t.lower() for x in ["chat", "featured", "verified", "call"]):
                                title = t
                                break
                if not title or len(title) < 3:
                    continue

                # Price
                price = ""
                for el in item.select("span, div"):
                    t = el.get_text(strip=True)
                    if "EGP" in t:
                        price = extract_price(t)
                        if price:
                            break

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
                row = make_row(lid, product_url, "olx_egypt", title, title, price, model, img_paths, "OLX Seller", "")
                append_row(row)
                update_prices(model, price)
                rows.append(row)
                print(f"  [OLX] {len(rows)}/{target}: {title[:55]}")

            delay()
        except Exception as e:
            print(f"  [OLX] error: {e}")
            delay(3, 7)

    print(f"[OLX Egypt] done — {len(rows)} rows")
    return rows


def scrape_opensooq(session, existing_ids, target=150):
    """
    OpenSooq Egypt — classifieds site, server-side rendered.
    Mobile phones category: https://eg.opensooq.com/en/mobile-phones
    """
    print(f"\n[OpenSooq] target={target}")
    rows = []
    seen = set(existing_ids)
    os_session = get_session(OPENSOOQ_HEADERS)

    for page in range(1, 30):
        if len(rows) >= target:
            break
        url = f"https://eg.opensooq.com/en/mobile-phones?page={page}"
        try:
            resp = os_session.get(url, timeout=25)
            if resp.status_code != 200:
                print(f"  [OpenSooq] page {page} → {resp.status_code}")
                delay(3, 7)
                continue
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.text, "html.parser")

            # OpenSooq listing cards
            items = (soup.select("li.post-cell") or
                     soup.select(".post-cell") or
                     soup.select("article.post") or
                     soup.select("[class*='post-item']") or
                     soup.select("li[class*='cell']"))

            if not items:
                # fallback: grab all product links
                items = soup.select("a[href*='/en/mobile-phones/']")
                items = [i for i in items if i.select_one("img")]

            if not items:
                print(f"  [OpenSooq] page {page} → no items, stopping")
                break

            print(f"  [OpenSooq] page {page} → {len(items)} items")

            for item in items:
                if len(rows) >= target:
                    break

                # URL
                link_el = item if item.name == "a" else item.select_one("a")
                if not link_el:
                    continue
                href = link_el.get("href", "")
                if not href:
                    continue
                product_url = urljoin("https://eg.opensooq.com", href)
                lid = generate_id(product_url)
                if lid in seen:
                    continue
                seen.add(lid)

                # Title
                title_el = (item.select_one("h2") or
                            item.select_one("h3") or
                            item.select_one("[class*='title']") or
                            item.select_one("[class*='name']") or
                            item.select_one("p"))
                title = title_el.get_text(strip=True) if title_el else ""
                if not title:
                    img = item.select_one("img")
                    title = img.get("alt", "") if img else ""
                if not title or len(title) < 3:
                    continue

                # Skip garbage titles (locations, generic phrases)
                skip_words = ["محافظة", "منطقة", "قطع غيار", "اكسسوار", "شاشة", "غطاء"]
                if any(w in title for w in skip_words) and len(title) < 15:
                    continue

                # Price
                price_el = (item.select_one("[class*='price']") or
                            item.select_one("[class*='Price']") or
                            item.select_one("span[class*='money']"))
                price = extract_price(price_el.get_text() if price_el else "")

                # Image
                img_el = item.select_one("img")
                img_url = ""
                if img_el:
                    img_url = (img_el.get("data-src", "") or
                               img_el.get("src", "") or
                               img_el.get("data-lazy", ""))
                img_paths = []
                if img_url and img_url.startswith("http"):
                    saved = save_image(img_url, lid, 0, os_session)
                    if saved:
                        img_paths.append(saved)

                # Seller
                seller_el = item.select_one("[class*='seller']") or item.select_one("[class*='user']")
                seller = seller_el.get_text(strip=True) if seller_el else "OpenSooq Seller"

                model = extract_phone_model(title)
                row = make_row(lid, product_url, "opensooq", title, title, price, model, img_paths, seller, "")
                append_row(row)
                update_prices(model, price)
                rows.append(row)
                print(f"  [OpenSooq] {len(rows)}/{target}: {title[:55]}")

            delay()
        except Exception as e:
            print(f"  [OpenSooq] error: {e}")
            delay(3, 7)

    print(f"[OpenSooq] done — {len(rows)} rows")
    return rows


# ── prices ────────────────────────────────────────────────────────────────────

def save_prices():
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
    print(f"\n[Prices] Updated — {len(existing)} models in prices.csv")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    init_csv()
    session = get_session()
    existing_ids = get_existing_ids()
    print(f"Existing rows in CSV: {len(existing_ids)}")

    all_rows = []
    all_rows.extend(scrape_olx_egypt(session, existing_ids, target=150))
    existing_ids = get_existing_ids()
    all_rows.extend(scrape_opensooq(session, existing_ids, target=150))

    save_prices()

    print(f"\n{'='*55}")
    print(f"SCAMMY SCRAPING COMPLETE")
    print(f"New rows this run : {len(all_rows)}")
    print(f"CSV               : {CSV_PATH}")
    print(f"Images            : {IMG_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
