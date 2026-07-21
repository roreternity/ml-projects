"""Сбор вакансий Python-разработчика через API HH.ru, сохранение в Excel."""
import re
import requests
import pandas as pd


def clean_html(raw: str) -> str:
    """Убирает HTML-теги из описания вакансии и обрезает до 300 символов."""
    return re.sub(r"<[^>]+>", " ", raw).strip()[:300]


def format_salary(salary: dict | None) -> str:
    if not salary:
        return "Не указана"
    lo = salary.get("from")
    hi = salary.get("to")
    cur = salary.get("currency", "")
    parts = []
    if lo:
        parts.append(f"от {lo}")
    if hi:
        parts.append(f"до {hi}")
    return " ".join(parts) + f" {cur}"


def fetch_vacancies(
    query: str = "Python разработчик",
    area: int = 1,  # 1 = Москва
    pages: int = 3,
) -> list[dict]:
    rows = []
    session = requests.Session()
    session.headers["User-Agent"] = "PracticalWork/1.0"

    for page in range(pages):
        resp = session.get(
            "https://api.hh.ru/vacancies",
            params={"text": query, "area": area, "per_page": 20, "page": page},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            break

        for item in items:
            # Список вакансий не содержит полного описания — берём его отдельным запросом
            detail = session.get(f"https://api.hh.ru/vacancies/{item['id']}").json()
            rows.append({
                "Название вакансии": item["name"],
                "Краткое описание": clean_html(detail.get("description", "")),
                "Зарплата": format_salary(item.get("salary")),
                "Ключевые навыки": ", ".join(s["name"] for s in detail.get("key_skills", [])),
            })
    return rows


if __name__ == "__main__":
    data = fetch_vacancies()
    df = pd.DataFrame(data)
    df.to_excel("hh_vacancies.xlsx", index=False)
    print(f"Сохранено {len(df)} вакансий -> hh_vacancies.xlsx")
