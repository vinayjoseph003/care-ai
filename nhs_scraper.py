import requests
from bs4 import BeautifulSoup
import json
import time

BASE_URL = "https://www.nhs.uk/conditions"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_condition_links():
    res = requests.get(BASE_URL, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if href.startswith("/conditions/") and href.count("/") <= 3:
            full_url = "https://www.nhs.uk" + href
            links.append(full_url)

    return list(set(links))


def scrape_condition(url):
    try:
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, "html.parser")

        title = soup.find("h1").get_text(strip=True)

        paragraphs = soup.select("main p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs[:10])

        return {
            "title": title,
            "text": text,
            "url": url
        }

    except Exception:
        return None


def main():
    print("🌐 Scraping NHS website...")
    links = get_condition_links()
    print(f"Found {len(links)} conditions")

    data = []
    for i, link in enumerate(links):
        print(f"[{i+1}/{len(links)}] {link}")
        item = scrape_condition(link)
        if item:
            data.append(item)
        time.sleep(0.5)

    with open("data/nhs_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Saved {len(data)} conditions")


if __name__ == "__main__":
    main()