"""Сбор карточек товаров через поисковое API Wildberries, сохранение в Excel."""
import requests
import pandas as pd


def fetch_wb_products(query: str = "беспроводные наушники", limit: int = 20) -> list[dict]:
    url = "https://search.wb.ru/exactmatch/ru/common/v5/search"
    params = {
        "query": query,
        "resultset": "catalog",
        "limit": limit,
        "sort": "popular",
        "page": 1,
        "appType": 1,
        "curr": "rub",
        "dest": -1257786,  # Москва
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    }
    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    products = resp.json().get("data", {}).get("products", [])

    rows = []
    for item in products:
        nm = item["id"]
        # Шаблон CDN Wildberries для превью — номер корзины/тома/партии
        # вычисляется из id товара по фиксированной формуле
        vol = nm // 100_000
        part = nm // 1_000
        img = (
            f"https://basket-{(vol % 13) + 1:02d}.wbbasket.ru"
            f"/vol{vol}/part{part}/{nm}/images/small/1.jpg"
        )
        rows.append({
            "Название товара": item.get("name", ""),
            "Бренд": item.get("brand", ""),
            "URL фотографии": img,
            "Цена со скидкой": item.get("salePriceU", 0) / 100,
            "Средняя оценка": item.get("reviewRating", 0),
            "Количество отзывов": item.get("feedbacks", 0),
        })
    return rows


if __name__ == "__main__":
    data = fetch_wb_products("беспроводные наушники")
    df = pd.DataFrame(data)
    df.to_excel("wb_competitors.xlsx", index=False)
    print(f"Сохранено {len(df)} товаров -> wb_competitors.xlsx")
