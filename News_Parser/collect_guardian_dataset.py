from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import certifi
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

API_URL = "https://content.guardianapis.com/search"
MAX_PAGE_SIZE = 50


# Тональность считаем простым методом — подсчётом позитивных/негативных слов
# в тексте статьи. Базовый подход без сложных NLP-моделей, но прозрачный
POSITIVE_WORDS = {
    "advance",
    "benefit",
    "best",
    "better",
    "boost",
    "breakthrough",
    "celebrate",
    "gain",
    "good",
    "great",
    "growth",
    "hope",
    "improve",
    "success",
    "support",
    "win",
}

NEGATIVE_WORDS = {
    "attack",
    "bad",
    "crisis",
    "damage",
    "death",
    "decline",
    "fail",
    "fear",
    "harm",
    "loss",
    "risk",
    "scandal",
    "threat",
    "war",
    "worse",
    "worst",
}


class LinkParser(HTMLParser):
    """Извлекает href-ссылки из HTML-тела статьи."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self.links.append(href)


@dataclass
class ArticleMetrics:
    """Одна обработанная статья — все поля итогового TSV-датасета."""

    date: str
    topic: str
    title: str
    author: str
    entities: str
    word_count: int
    links: str
    readability_score: float | None
    readability_level: str
    sentiment: str
    sentiment_score: int
    url: str


def build_api_url(api_key: str, page: int, page_size: int, section: str | None) -> str:
    params = {
        "api-key": api_key,
        "page": str(page),
        "page-size": str(page_size),
        "order-by": "newest",
        "show-fields": "headline,byline,body,bodyText,trailText,wordcount",
        "show-tags": "keyword",
        "type": "article",
    }
    if section:
        params["section"] = section

    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def extract_api_key(value: str) -> str:
    """Позволяет передать в GUARDIAN_API_KEY как сам ключ, так и полную ссылку с ним."""
    if value.startswith("http"):
        parsed = urllib.parse.urlparse(value)
        query = urllib.parse.parse_qs(parsed.query)
        return query.get("api-key", [""])[0]
    return value


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "guardian-metrics-student/1.0"},
    )

    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Guardian API вернул HTTP {error.code}: {message}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Не удалось подключиться к Guardian API: {error.reason}") from error


def normalize_link(link: str) -> str | None:
    """Отбрасывает query-параметры и оставляет только валидные http(s)-ссылки."""
    link = html.unescape(link)
    parsed = urllib.parse.urlparse(link)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    return urllib.parse.urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, "")
    )


def extract_links(body_html: str) -> list[str]:
    parser = LinkParser()
    parser.feed(body_html)

    links = []
    for link in parser.links:
        normalized = normalize_link(link)
        if normalized:
            links.append(normalized)

    return sorted(set(links))


def clean_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text))


def count_sentences(text: str) -> int:
    sentences = re.findall(r"[^.!?]+[.!?]", text)
    return max(1, len(sentences))


def count_syllables(word: str) -> int:
    """Грубая оценка числа слогов — считает группы гласных подряд."""
    word = word.lower()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    groups = re.findall(r"[aeiouy]+", word)
    syllables = len(groups)
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def readability_score(text: str) -> float | None:
    """Индекс читаемости Flesch-Kincaid Grade Level.
    Чем выше значение — тем сложнее текст (примерно соответствует классу школы США).
    """
    words = re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text)
    if not words:
        return None

    word_total = len(words)
    sentence_total = count_sentences(text)
    syllable_total = sum(count_syllables(word) for word in words)
    grade = 0.39 * (word_total / sentence_total) + 11.8 * (syllable_total / word_total) - 15.59
    return round(grade, 2)


def readability_level(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score < 6:
        return "easy"
    if score < 10:
        return "medium"
    if score < 14:
        return "hard"
    return "very hard"


def sentiment_score(text: str) -> int:
    words = [word.lower() for word in re.findall(r"\b[A-Za-z]+\b", text)]
    positive_count = sum(1 for word in words if word in POSITIVE_WORDS)
    negative_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    return positive_count - negative_count


def sentiment_label(score: int) -> str:
    if score > 1:
        return "positive"
    if score < -1:
        return "negative"
    return "neutral"


def tags_to_entities(tags: list[dict[str, Any]]) -> str:
    """Собирает ключевые сущности статьи из тегов типа keyword."""
    entity_names = []
    for tag in tags:
        if tag.get("type") == "keyword":
            name = tag.get("webTitle")
            if name:
                entity_names.append(name)
    return "; ".join(sorted(set(entity_names)))


def article_to_metrics(article: dict[str, Any]) -> ArticleMetrics | None:
    fields = article.get("fields", {})
    body_text = clean_text(fields.get("bodyText", ""))
    if not body_text:
        return None  # статьи без текста тела (например, только видео) пропускаем

    links = extract_links(fields.get("body", ""))
    word_count = int(fields.get("wordcount") or count_words(body_text))
    reading_score = readability_score(body_text)
    tone_score = sentiment_score(body_text)

    return ArticleMetrics(
        date=article.get("webPublicationDate", ""),
        topic=article.get("sectionName", ""),
        title=fields.get("headline") or article.get("webTitle", ""),
        author=fields.get("byline", ""),
        entities=tags_to_entities(article.get("tags", [])),
        word_count=word_count,
        links="; ".join(links),
        readability_score=reading_score,
        readability_level=readability_level(reading_score),
        sentiment=sentiment_label(tone_score),
        sentiment_score=tone_score,
        url=article.get("webUrl", ""),
    )


def collect_articles(api_key: str, limit: int, section: str | None, pause: float) -> list[ArticleMetrics]:
    """Постранично собирает статьи, пропуская дубликаты по URL и статьи без текста."""
    collected: list[ArticleMetrics] = []
    seen_urls: set[str] = set()
    page = 1

    while len(collected) < limit:
        url = build_api_url(api_key, page, MAX_PAGE_SIZE, section)
        data = fetch_json(url)
        response = data.get("response", {})
        articles = response.get("results", [])

        if not articles:
            break

        for article in articles:
            article_url = article.get("webUrl", "")
            if article_url in seen_urls:
                continue

            metrics = article_to_metrics(article)
            if metrics is None:
                continue

            seen_urls.add(article_url)
            collected.append(metrics)

            if len(collected) >= limit:
                break

        total_pages = int(response.get("pages", page))
        print(f"страница {page}/{total_pages}: собрано {len(collected)}/{limit}", file=sys.stderr)
        if page >= total_pages:
            break

        page += 1
        time.sleep(pause)  # пауза между запросами — не перегружаем API

    return collected


def write_tsv(path: str, rows: list[ArticleMetrics]) -> None:
    fieldnames = [
        "date",
        "topic",
        "title",
        "author",
        "entities",
        "word_count",
        "links",
        "readability_score",
        "readability_level",
        "sentiment",
        "sentiment_score",
        "url",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Экспорт метрик статей Guardian в TSV.")
    parser.add_argument("--limit", type=int, default=100, help="Сколько валидных статей собрать.")
    parser.add_argument("--section", help="Раздел Guardian, например: world, politics, sport.")
    parser.add_argument("--output", default="guardian_news.tsv", help="Путь к выходному TSV-файлу.")
    parser.add_argument("--pause", type=float, default=1.1, help="Пауза между запросами к API, сек.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = extract_api_key(os.getenv("GUARDIAN_API_KEY") or "")

    if not api_key:
        print("Задайте переменную окружения GUARDIAN_API_KEY перед запуском.", file=sys.stderr)
        return 1

    try:
        rows = collect_articles(
            api_key=api_key,
            limit=args.limit,
            section=args.section,
            pause=args.pause,
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    write_tsv(args.output, rows)
    print(f"сохранено {len(rows)} строк в {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
