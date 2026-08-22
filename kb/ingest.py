import os
import re
import time
import uuid
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# =============================================================================
# Configuration
# =============================================================================

SITEMAP_INDEX = os.environ.get("SITEMAP_INDEX", "").strip()
SITE = os.environ.get("SITE", "").strip().lower()
COLLECTION = os.environ.get("COLLECTION", "").strip()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")

CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "1400"))
OVERLAP_CHARS = int(os.environ.get("OVERLAP_CHARS", "160"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "50"))
MAX_CHUNKS_PER_PAGE = int(os.environ.get("MAX_CHUNKS_PER_PAGE", "150"))

INCREMENTAL = os.environ.get("INCREMENTAL", "1").lower() not in (
    "0", "false", "no", "off"
)

PRUNE = os.environ.get("PRUNE", "1").lower() not in (
    "0", "false", "no", "off"
)

# Important for first run with this new state format:
# Existing Qdrant URLs are treated as an already indexed baseline.
BOOTSTRAP_EXISTING = os.environ.get("BOOTSTRAP_EXISTING", "1").lower() not in (
    "0", "false", "no", "off"
)

STATE_DIR = Path(os.environ.get("STATE_DIR", "/app/state"))

REQUEST_HEADERS = {
    "User-Agent": "CoinchargeKBIngest/2.0 (+https://coincharge.io)"
}

SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico",
    ".tif", ".tiff", ".pdf", ".zip", ".rar", ".7z", ".gz", ".tar",
    ".mp3", ".mp4", ".avi", ".mov", ".webm",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".xml",
)

SKIP_PATH_CONTAINS = (
    "/wp-content/uploads/",
    "/wp-json/",
    "/feed/",
)

SKIP_PATH_PREFIXES = (
    "/category/",
    "/tag/",
    "/author/",
    "/en/category/",
    "/en/tag/",
    "/en/author/",
)

SKIP_EXACT_PATHS = (
    "/cart/",
    "/checkout/",
    "/checkout-2/",
    "/my-account/",
    "/suche/",
    "/sample-page/",
    "/en/cart/",
    "/en/checkout/",
    "/en/checkout-2/",
    "/en/my-account/",
    "/en/sample-page/",
)


# =============================================================================
# Generic helpers
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def fetch_response(url: str) -> Optional[requests.Response]:
    r = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=30,
        allow_redirects=True,
    )

    if r.status_code in (404, 410):
        return None

    r.raise_for_status()
    return r


def fetch(url: str) -> Optional[bytes]:
    r = fetch_response(url)
    if r is None:
        return None
    return r.content


def fetch_text(url: str) -> Optional[str]:
    content = fetch(url)
    if content is None:
        return None
    return content.decode("utf-8", errors="replace")


def clean_title(raw_title: str) -> str:
    title = normalize_whitespace(raw_title)

    if not title:
        return ""

    for sep in (" | ", " – ", " — ", " - "):
        parts = [p.strip() for p in title.split(sep)]

        if len(parts) <= 1:
            continue

        suffix = parts[-1].lower()

        if "coincharge" in suffix:
            return parts[0]

        if "coinsnap" in suffix:
            return parts[0]

        if "coinpages" in suffix:
            return parts[0]

    return title


# =============================================================================
# Language detection
# =============================================================================

def detect_lang(soup: BeautifulSoup, url: str, site: str) -> str:
    html_tag = soup.find("html")
    lang_attr = ""

    if html_tag and html_tag.get("lang"):
        lang_attr = str(html_tag.get("lang")).strip().lower()

    # Prefer explicit HTML language.
    if lang_attr:
        m = re.match(r"^([a-z]{2})", lang_attr)
        if m:
            return m.group(1)

    parsed = urlparse(url)
    path = parsed.path or "/"

    # Coincharge: German root, English under /en/
    if "coincharge.io" in site:
        if re.match(r"^/en(/|$)", path, flags=re.IGNORECASE):
            return "en"
        return "de"

    # Coinpages currently German.
    if "coinpages.io" in site:
        return "de"

    # Coinsnap main site:
    # current root is English; future /de/, /fr/, etc. are supported.
    if site in ("coinsnap.io", "www.coinsnap.io"):
        m = re.match(r"^/([a-z]{2})(/|$)", path, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return "en"

    # Coinsnap developer documentation:
    # currently English, future language folders are supported as well.
    if "docs.coinsnap.io" in site:
        m = re.match(r"^/([a-z]{2})(/|$)", path, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return "en"

    return "en"


# =============================================================================
# Coinpages special handling
# =============================================================================

def is_coinpages_site(site: str) -> bool:
    return "coinpages.io" in (site or "")


def is_coinpages_place_url(url: str) -> bool:
    parsed = urlparse(url)
    path = (parsed.path or "").lower()
    return path.startswith("/listings/")


def _extract_field_line(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}\s*:\s*(.+)"
    m = re.search(pattern, text, flags=re.IGNORECASE)

    if not m:
        return ""

    value = m.group(1).strip()
    value = re.split(r"\n", value)[0].strip()

    return normalize_whitespace(value)


def parse_coinpages_category(text: str, soup: BeautifulSoup) -> str:
    value = _extract_field_line(text, "Kategorie")
    if value:
        return value

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(" ", strip=True) or ""

        if not raw:
            continue

        m = re.search(
            r'"articleSection"\s*:\s*"([^"]+)"',
            raw,
            flags=re.IGNORECASE,
        )

        if m:
            return normalize_whitespace(m.group(1))

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if "/listings/category/" in href:
            label = a.get_text(" ", strip=True)

            if label:
                return normalize_whitespace(label)

    m = re.search(
        r"/listings/category/([^/]+)/",
        str(soup),
        flags=re.IGNORECASE,
    )

    if m:
        slug = m.group(1).replace("-", " ")
        return normalize_whitespace(slug)

    return ""


def parse_coinpages_email(text: str) -> str:
    value = _extract_field_line(text, "Email Adresse")

    if value:
        return value

    m = re.search(
        r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}",
        text,
        flags=re.IGNORECASE,
    )

    return m.group(0) if m else ""


def parse_coinpages_phone(text: str) -> str:
    value = _extract_field_line(text, "telefon")
    if value:
        return value

    value = _extract_field_line(text, "Telefon")
    if value:
        return value

    return ""


def parse_coinpages_address(text: str) -> str:
    value = _extract_field_line(text, "Adresse")
    if value:
        return value

    value = _extract_field_line(text, "Anschrift")
    if value:
        return value

    return ""


def parse_coinpages_website(text: str, soup: BeautifulSoup) -> str:
    value = _extract_field_line(text, "Webseite")
    if value:
        return value

    value = _extract_field_line(text, "Website")
    if value:
        return value

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if href.startswith("http") and "coinpages.io" not in href:
            return href

    return ""


def parse_coinpages_payment_methods(text: str) -> List[str]:
    value = _extract_field_line(text, "Bezahlmethoden")
    source = value if value else text

    methods: List[str] = []
    lower = source.lower()

    if "bitcoin lightning" in lower or "lightning" in lower:
        methods.append("lightning")

    if "bitcoin" in lower:
        methods.append("bitcoin")

    out: List[str] = []
    seen: Set[str] = set()

    for method in methods:
        if method in seen:
            continue

        seen.add(method)
        out.append(method)

    return out


def parse_coinpages_opening_hours(text: str) -> str:
    value = _extract_field_line(text, "Öffnungszeiten")

    if value:
        return value

    m = re.search(r"Öffnungszeiten\s*(.+)", text, flags=re.IGNORECASE)

    if m:
        return normalize_whitespace(m.group(1))

    return ""


def parse_coinpages_city_country_from_url(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.strip("/").split("/") if p]

    # listings/deutschland/hessen/frankfurt/italia-90
    if len(parts) >= 5 and parts[0] == "listings":
        country = normalize_whitespace(parts[1].replace("-", " "))
        city = normalize_whitespace(parts[3].replace("-", " "))
        return city, country

    return "", ""


# =============================================================================
# Structured section extraction
# =============================================================================

SECTION_CHUNK_SCHEMA_VERSION = 3


def _clean_section_text(value: str) -> str:
    value = str(value or "").replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def extract_structured_sections(
    content_root: Tag,
    page_title: str,
) -> List[Dict[str, Any]]:
    """
    Preserve H1/H2/H3/H4 hierarchy.

    Generic by design:
    no knowledge of Coinsnap, Bringin, DFX,
    Wave.Space or other specific entities.
    """

    heading_levels = {
        "h1": 1,
        "h2": 2,
        "h3": 3,
        "h4": 4,
    }

    heading_stack: Dict[int, str] = {}

    sections: List[Dict[str, Any]] = []

    current_title = (
        _clean_section_text(page_title)
        or "Page"
    )

    current_path: List[str] = [
        current_title
    ]

    current_parts: List[str] = []


    def flush_current() -> None:
        nonlocal current_parts

        cleaned_parts: List[str] = []

        for value in current_parts:
            value = _clean_section_text(value)

            if not value:
                continue

            if (
                cleaned_parts
                and cleaned_parts[-1] == value
            ):
                continue

            cleaned_parts.append(value)

        body = "\n".join(
            cleaned_parts
        ).strip()

        if body:
            sections.append(
                {
                    "section_title": (
                        current_title
                        or page_title
                        or "Page"
                    ),
                    "section_path": list(
                        current_path
                    ),
                    "text": body,
                }
            )

        current_parts = []


    for node in content_root.descendants:

        if (
            isinstance(node, Tag)
            and node.name in heading_levels
        ):
            flush_current()

            heading = _clean_section_text(
                node.get_text(
                    " ",
                    strip=True,
                )
            )

            level = heading_levels[
                node.name
            ]

            for existing_level in list(
                heading_stack.keys()
            ):
                if existing_level >= level:
                    del heading_stack[
                        existing_level
                    ]

            if heading:
                heading_stack[level] = heading

            path_values = [
                heading_stack[key]
                for key in sorted(
                    heading_stack.keys()
                )
                if heading_stack.get(key)
            ]

            current_title = (
                heading
                or page_title
                or "Page"
            )

            current_path = (
                path_values
                or [current_title]
            )

            continue


        if not isinstance(
            node,
            NavigableString,
        ):
            continue

        parent = getattr(
            node,
            "parent",
            None,
        )

        if parent is None:
            continue

        # Heading content belongs to section metadata,
        # not to the section body.
        if (
            getattr(parent, "name", None)
            in heading_levels
        ):
            continue

        try:
            heading_parent = parent.find_parent(
                list(
                    heading_levels.keys()
                )
            )
        except Exception:
            heading_parent = None

        if heading_parent is not None:
            continue

        value = _clean_section_text(
            str(node)
        )

        if not value:
            continue

        current_parts.append(
            value
        )


    flush_current()


    # Fallback for pages without usable headings.
    if not sections:
        fallback = _clean_section_text(
            content_root.get_text(
                "\n",
                strip=True,
            )
        )

        if fallback:
            sections.append(
                {
                    "section_title": (
                        page_title
                        or "Page"
                    ),
                    "section_path": [
                        page_title
                        or "Page"
                    ],
                    "text": fallback,
                }
            )


    return sections


# =============================================================================
# Page extraction
# =============================================================================

def extract_page_content(
    html: bytes,
    url: str,
    site: str,
) -> Tuple[str, str, str, Dict[str, Any]]:

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "picture",
        "source",
    ]):
        tag.decompose()

    # Remove typical navigation/noise.
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()

    lang = detect_lang(soup, url, site)

    og_title = soup.find("meta", attrs={"property": "og:title"})

    og_image = soup.find(
        "meta",
        attrs={"property": "og:image"},
    )

    twitter_image = soup.find(
        "meta",
        attrs={"name": "twitter:image"},
    )

    image_url = ""

    if og_image and og_image.get("content"):
        image_url = urljoin(
            url,
            str(og_image.get("content")).strip(),
        )

    elif twitter_image and twitter_image.get("content"):
        image_url = urljoin(
            url,
            str(twitter_image.get("content")).strip(),
        )

    h1_tag = soup.find("h1")
    title_tag = soup.find("title")

    if og_title and og_title.get("content"):
        raw_title = str(og_title.get("content"))
    elif h1_tag and h1_tag.get_text(strip=True):
        raw_title = h1_tag.get_text(" ", strip=True)
    elif title_tag and title_tag.get_text(strip=True):
        raw_title = title_tag.get_text(" ", strip=True)
    else:
        raw_title = url

    title = clean_title(raw_title) or url

    content_root = (
        soup.find("main")
        or soup.find("article")
        or soup.body
        or soup
    )

    sections = extract_structured_sections(
        content_root,
        title,
    )

    text = content_root.get_text("\n", strip=True)

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    meta: Dict[str, Any] = {
        "doc_type": "content",
        "city": "",
        "country": "",
        "category": "",
        "payment_methods": [],
        "opening_hours": "",
        "website": "",
        "address": "",
        "email": "",
        "phone": "",
        "image": image_url,
        "sections": sections,
    }

    # Developer documentation gets its own document type.
    if "docs.coinsnap.io" in site:
        meta["doc_type"] = "developer_docs"

    if is_coinpages_site(site):
        if is_coinpages_place_url(url):
            city, country = parse_coinpages_city_country_from_url(url)

            meta["doc_type"] = "place"
            meta["city"] = city
            meta["country"] = country
            meta["category"] = parse_coinpages_category(text, soup)
            meta["payment_methods"] = parse_coinpages_payment_methods(text)
            meta["opening_hours"] = parse_coinpages_opening_hours(text)
            meta["website"] = parse_coinpages_website(text, soup)
            meta["address"] = parse_coinpages_address(text)
            meta["email"] = parse_coinpages_email(text)
            meta["phone"] = parse_coinpages_phone(text)
        else:
            meta["doc_type"] = "content"

    return title, text, lang, meta


# =============================================================================
# Chunking / embeddings
# =============================================================================

def chunk_text(text: str) -> List[str]:
    if len(text) <= CHUNK_CHARS:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        chunks.append(text[start:end])

        if end >= len(text):
            break

        next_start = end - OVERLAP_CHARS

        # Protection against pathological configuration.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def build_section_chunks(
    text: str,
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Create chunks inside semantic HTML sections.

    A chunk never crosses a section boundary.

    The existing CHUNK_CHARS and OVERLAP_CHARS settings
    are retained inside each individual section.
    """

    result: List[Dict[str, Any]] = []

    sections = (
        meta.get("sections")
        or []
    )

    for section in sections:

        if not isinstance(
            section,
            dict,
        ):
            continue

        section_text = str(
            section.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not section_text:
            continue

        section_title = str(
            section.get(
                "section_title",
                "",
            )
            or ""
        ).strip()

        raw_path = (
            section.get(
                "section_path",
                []
            )
            or []
        )

        section_path = [
            str(value).strip()
            for value in raw_path
            if str(value).strip()
        ]

        pieces = chunk_text(
            section_text
        )

        for piece in pieces:

            piece = str(
                piece or ""
            ).strip()

            if not piece:
                continue

            result.append(
                {
                    "text": piece,
                    "section_title": (
                        section_title
                    ),
                    "section_path": (
                        section_path
                    ),
                }
            )


    # Backwards-compatible fallback for pages whose HTML
    # does not contain useful headings.
    if not result:

        for piece in chunk_text(
            text
        ):

            piece = str(
                piece or ""
            ).strip()

            if not piece:
                continue

            result.append(
                {
                    "text": piece,
                    "section_title": "",
                    "section_path": [],
                }
            )

    return result


def build_embedding_inputs(
    title: str,
    lang: str,
    chunks: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> List[str]:

    inputs: List[str] = []

    for chunk in chunks:

        parts: List[str] = []

        if title:
            parts.append(
                f"Title: {title}"
            )

        parts.append(
            f"Language: {lang}"
        )

        if meta.get("doc_type"):
            parts.append(
                f"Document-Type: {meta['doc_type']}"
            )

        if meta.get("city"):
            parts.append(
                f"City: {meta['city']}"
            )

        if meta.get("country"):
            parts.append(
                f"Country: {meta['country']}"
            )

        if meta.get("category"):
            parts.append(
                f"Category: {meta['category']}"
            )

        if meta.get("payment_methods"):
            parts.append(
                "Payment-Methods: "
                + ", ".join(
                    meta["payment_methods"]
                )
            )

        section_title = str(
            chunk.get(
                "section_title",
                "",
            )
            or ""
        ).strip()

        section_path = [
            str(value).strip()
            for value in (
                chunk.get(
                    "section_path",
                    []
                )
                or []
            )
            if str(value).strip()
        ]

        if section_title:
            parts.append(
                f"Section: {section_title}"
            )

        if section_path:
            parts.append(
                "Section-Path: "
                + " > ".join(
                    section_path
                )
            )

        chunk_text_value = str(
            chunk.get(
                "text",
                "",
            )
            or ""
        ).strip()

        parts.append(
            chunk_text_value
        )

        inputs.append(
            "\n".join(parts)
        )

    return inputs

def embed(client: OpenAI, texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]

        res = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch,
        )

        vectors.extend([item.embedding for item in res.data])

    return vectors


# =============================================================================
# Content hashes
# =============================================================================

def canonical_hash_payload(
    title: str,
    text: str,
    lang: str,
    meta: Dict[str, Any],
) -> str:

    relevant = {
        "title": title,
        "text": text,
        "lang": lang,
        "doc_type": meta.get("doc_type", "content"),
        "city": meta.get("city", ""),
        "country": meta.get("country", ""),
        "category": meta.get("category", ""),
        "payment_methods": meta.get("payment_methods", []),
        "opening_hours": meta.get("opening_hours", ""),
        "website": meta.get("website", ""),
        "address": meta.get("address", ""),
        "email": meta.get("email", ""),
        "phone": meta.get("phone", ""),
        "image": meta.get("image", ""),
        "chunk_schema_version": (
            SECTION_CHUNK_SCHEMA_VERSION
        ),
        "section_structure": [
            {
                "section_title": (
                    section.get(
                        "section_title",
                        "",
                    )
                ),
                "section_path": (
                    section.get(
                        "section_path",
                        [],
                    )
                ),
            }
            for section in (
                meta.get("sections")
                or []
            )
            if isinstance(
                section,
                dict,
            )
        ],
    }

    return json.dumps(
        relevant,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def calculate_content_hash(
    title: str,
    text: str,
    lang: str,
    meta: Dict[str, Any],
) -> str:

    raw = canonical_hash_payload(title, text, lang, meta)

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


# =============================================================================
# State
# =============================================================================

def state_file_path() -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / f"{COLLECTION}.json"


def new_empty_state() -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "site": SITE,
        "collection": COLLECTION,
        "sitemap_index": SITEMAP_INDEX,
        "embed_model": EMBED_MODEL,
        "last_run": None,
        "last_run_status": "never",
        "pages": {},
    }


def load_state() -> Tuple[Dict[str, Any], bool]:
    path = state_file_path()

    if not path.exists():
        return new_empty_state(), True

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Old state formats are deliberately not trusted for hashes.
        if (
            not isinstance(raw, dict)
            or raw.get("schema_version") != 2
            or not isinstance(raw.get("pages"), dict)
        ):
            print(
                f"[state] legacy state detected: {path}; "
                "creating v2 baseline"
            )
            return new_empty_state(), True

        raw["site"] = SITE
        raw["collection"] = COLLECTION
        raw["sitemap_index"] = SITEMAP_INDEX
        raw["embed_model"] = EMBED_MODEL

        return raw, False

    except Exception as e:
        print(f"[state] could not read {path}: {e}")
        print("[state] starting with a fresh v2 state")
        return new_empty_state(), True


def save_state(state: Dict[str, Any]) -> None:
    path = state_file_path()
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    os.replace(tmp, path)


# =============================================================================
# Qdrant
# =============================================================================

def collection_exists(qd: QdrantClient, collection: str) -> bool:
    names = {
        c.name
        for c in qd.get_collections().collections
    }

    return collection in names


def ensure_collection(
    qd: QdrantClient,
    collection: str,
    size: int,
) -> None:

    if collection_exists(qd, collection):
        return

    print(
        f"[qdrant] creating collection={collection} "
        f"vector_size={size}"
    )

    qd.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(
            size=size,
            distance=qm.Distance.COSINE,
        ),
    )


def url_exists_in_collection(
    qd: QdrantClient,
    collection: str,
    url: str,
) -> bool:

    if not collection_exists(qd, collection):
        return False

    try:
        points, _ = qd.scroll(
            collection_name=collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="url",
                        match=qm.MatchValue(value=url),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )

        return bool(points)

    except Exception as e:
        print(
            f"[qdrant] existing-url check failed for {url}: {e}"
        )
        return False


def delete_existing_points_for_url(
    qd: QdrantClient,
    collection: str,
    url: str,
) -> None:

    if not collection_exists(qd, collection):
        return

    qd.delete(
        collection_name=collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="url",
                        match=qm.MatchValue(value=url),
                    )
                ]
            )
        ),
        wait=True,
    )


# =============================================================================
# Sitemap
# =============================================================================

def parse_xml_locs(xml_text: str) -> List[str]:
    soup = BeautifulSoup(xml_text, "xml")
    locs: List[str] = []

    for loc in soup.find_all("loc"):
        value = (loc.get_text() or "").strip()

        if value:
            locs.append(value)

    return locs


def should_skip_sitemap_url(
    sitemap_url: str,
    site: str,
) -> bool:

    s = sitemap_url.lower()

    if is_coinpages_site(site):
        if "gd_place_tags-sitemap" in s:
            return True

        if "gd_placecategory-sitemap" in s:
            return True

        if "gd_location_country-sitemap" in s:
            return True

        if s.endswith("/category-sitemap.xml"):
            return True

    return False


def should_index_url(url: str, site: str) -> bool:
    if not url:
        return False

    parsed = urlparse(url)
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    normalized_site = site.lower().replace("www.", "")

    normalized_netloc = netloc.replace("www.", "")

    if (
        normalized_site
        and normalized_site != normalized_netloc
    ):
        return False

    if path in SKIP_EXACT_PATHS:
        return False

    for marker in SKIP_PATH_CONTAINS:
        if marker in path:
            return False

    for prefix in SKIP_PATH_PREFIXES:
        if path.startswith(prefix):
            return False

    for ext in SKIP_EXTENSIONS:
        if path.endswith(ext):
            return False

    if is_coinpages_site(site):
        if (
            "/tag/" in path
            or "/category/" in path
            or "/author/" in path
        ):
            return False

    return True


def is_html_response(response: requests.Response) -> bool:
    content_type = (
        response.headers.get("Content-Type")
        or ""
    ).lower()

    return (
        "text/html" in content_type
        or "application/xhtml+xml" in content_type
    )


def expand_sitemap(
    start_url: str,
    site: str,
) -> List[str]:

    seen_sitemaps: Set[str] = set()
    collected_urls: List[str] = []

    def _walk(sitemap_url: str) -> None:
        if sitemap_url in seen_sitemaps:
            return

        if should_skip_sitemap_url(sitemap_url, site):
            print("skip sitemap:", sitemap_url)
            return

        seen_sitemaps.add(sitemap_url)

        print("read sitemap:", sitemap_url)

        xml_text = fetch_text(sitemap_url)

        if not xml_text:
            print("empty sitemap:", sitemap_url)
            return

        soup = BeautifulSoup(xml_text, "xml")

        if soup.find("sitemapindex"):
            child_sitemaps = parse_xml_locs(xml_text)

            print(
                "child sitemaps found:",
                len(child_sitemaps),
            )

            for child_url in child_sitemaps:
                _walk(child_url)

            return

        if soup.find("urlset"):
            page_urls = parse_xml_locs(xml_text)

            print(
                "page urls found in sitemap:",
                len(page_urls),
            )

            for url in page_urls:
                if should_index_url(url, site):
                    collected_urls.append(url)

            return

        locs = parse_xml_locs(xml_text)

        print("fallback locs found:", len(locs))

        for url in locs:
            if should_index_url(url, site):
                collected_urls.append(url)

    _walk(start_url)

    deduped: List[str] = []
    seen_urls: Set[str] = set()

    for url in collected_urls:
        if url in seen_urls:
            continue

        seen_urls.add(url)
        deduped.append(url)

    return deduped


# =============================================================================
# Index one page
# =============================================================================

def write_page_to_qdrant(
    oa: OpenAI,
    qd: QdrantClient,
    url: str,
    title: str,
    text: str,
    lang: str,
    meta: Dict[str, Any],
    content_hash: str,
) -> int:

    chunks = build_section_chunks(
        text,
        meta,
    )

    if len(chunks) > MAX_CHUNKS_PER_PAGE:
        raise ValueError(
            f"too many chunks: {len(chunks)} "
            f"(max {MAX_CHUNKS_PER_PAGE})"
        )

    embedding_inputs = build_embedding_inputs(
        title,
        lang,
        chunks,
        meta,
    )

    vectors = embed(oa, embedding_inputs)

    if not vectors:
        raise ValueError("no vectors returned")

    ensure_collection(
        qd,
        COLLECTION,
        len(vectors[0]),
    )

    # Only delete old data after new embeddings were generated
    # successfully. This avoids losing a working page when OpenAI fails.
    delete_existing_points_for_url(
        qd,
        COLLECTION,
        url,
    )

    points = []
    chunk_count = len(chunks)

    for i, vec in enumerate(vectors):
        payload = {
            "site": SITE,
            "url": url,
            "title": title,
            "lang": lang,
            "doc_type": meta.get("doc_type", "content"),
            "city": meta.get("city", ""),
            "country": meta.get("country", ""),
            "category": meta.get("category", ""),
            "payment_methods": meta.get(
                "payment_methods",
                [],
            ),
            "opening_hours": meta.get(
                "opening_hours",
                "",
            ),
            "website": meta.get("website", ""),
            "address": meta.get("address", ""),
            "email": meta.get("email", ""),
            "phone": meta.get("phone", ""),
            "image": meta.get("image", ""),
            "section_title": (
                chunks[i].get(
                    "section_title",
                    "",
                )
            ),
            "section_path": (
                chunks[i].get(
                    "section_path",
                    [],
                )
                or []
            ),
            "chunk_schema_version": (
                SECTION_CHUNK_SCHEMA_VERSION
            ),
            "chunk_index": i,
            "chunk_count": chunk_count,
            "content_hash": content_hash,
            "indexed_at": utc_now(),
            "text": (
                chunks[i].get(
                    "text",
                    "",
                )
            ),
        }

        points.append(
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            )
        )

    qd.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )

    return len(points)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not SITEMAP_INDEX:
        raise ValueError("SITEMAP_INDEX is not set")

    if not SITE:
        raise ValueError("SITE is not set")

    if not COLLECTION:
        raise ValueError("COLLECTION is not set")

    print("============================================================")
    print("Coincharge KB Ingest v2")
    print("============================================================")
    print("site:", SITE)
    print("collection:", COLLECTION)
    print("sitemap index:", SITEMAP_INDEX)
    print("embed model:", EMBED_MODEL)
    print("incremental:", INCREMENTAL)
    print("prune:", PRUNE)
    print("bootstrap existing:", BOOTSTRAP_EXISTING)
    print("state:", state_file_path())
    print("============================================================")

    oa = OpenAI(api_key=OPENAI_API_KEY)
    qd = QdrantClient(url=QDRANT_URL)

    state, needs_bootstrap = load_state()
    old_pages: Dict[str, Any] = state.get("pages", {})

    urls = expand_sitemap(
        SITEMAP_INDEX,
        SITE,
    )

    print("urls after sitemap expand:", len(urls))

    sitemap_url_set = set(urls)

    indexed_new = 0
    indexed_changed = 0
    unchanged = 0
    bootstrapped = 0
    skipped = 0
    pruned = 0
    page_errors = 0
    total_points_written = 0

    new_pages: Dict[str, Any] = dict(old_pages)

    for number, url in enumerate(urls, start=1):
        try:
            response = fetch_response(url)

            if response is None:
                print(
                    f"[{number}/{len(urls)}] "
                    f"skip missing: {url}"
                )
                skipped += 1
                continue

            if not is_html_response(response):
                print(
                    f"[{number}/{len(urls)}] "
                    f"skip non-html: {url} | "
                    f"{response.headers.get('Content-Type', 'unknown')}"
                )
                skipped += 1
                continue

            title, text, lang, meta = extract_page_content(
                response.content,
                url,
                SITE,
            )

            if len(text) < 120:
                print(
                    f"[{number}/{len(urls)}] "
                    f"skip too short: {url}"
                )
                skipped += 1
                continue

            chunks = chunk_text(text)

            if len(chunks) > MAX_CHUNKS_PER_PAGE:
                print(
                    f"[{number}/{len(urls)}] "
                    f"skip too many chunks: {url} | "
                    f"{len(chunks)}"
                )
                skipped += 1
                continue

            content_hash = calculate_content_hash(
                title,
                text,
                lang,
                meta,
            )

            old = old_pages.get(url)

            # Normal incremental path.
            if (
                INCREMENTAL
                and old
                and old.get("hash") == content_hash
            ):
                unchanged += 1

                old["last_seen"] = utc_now()
                old["title"] = title
                old["lang"] = lang
                old["doc_type"] = meta.get(
                    "doc_type",
                    "content",
                )
                old["chunk_count"] = len(chunks)

                new_pages[url] = old

                print(
                    f"[{number}/{len(urls)}] unchanged: "
                    f"{url}"
                )

                continue

            # First run with the new state format.
            #
            # Today's collections were already freshly rebuilt.
            # If Qdrant already contains this URL, adopt it as the
            # baseline without paying for another embedding call.
            if (
                INCREMENTAL
                and needs_bootstrap
                and BOOTSTRAP_EXISTING
                and url_exists_in_collection(
                    qd,
                    COLLECTION,
                    url,
                )
            ):
                bootstrapped += 1

                new_pages[url] = {
                    "hash": content_hash,
                    "title": title,
                    "lang": lang,
                    "doc_type": meta.get(
                        "doc_type",
                        "content",
                    ),
                    "chunk_count": len(chunks),
                    "last_seen": utc_now(),
                    "indexed_at": "baseline-existing-qdrant",
                }

                print(
                    f"[{number}/{len(urls)}] "
                    f"baseline existing: {url}"
                )

                continue

            is_change = bool(old)

            written = write_page_to_qdrant(
                oa=oa,
                qd=qd,
                url=url,
                title=title,
                text=text,
                lang=lang,
                meta=meta,
                content_hash=content_hash,
            )

            total_points_written += written

            if is_change:
                indexed_changed += 1
                action = "changed"
            else:
                indexed_new += 1
                action = "new"

            now = utc_now()

            new_pages[url] = {
                "hash": content_hash,
                "title": title,
                "lang": lang,
                "doc_type": meta.get(
                    "doc_type",
                    "content",
                ),
                "chunk_count": len(chunks),
                "last_seen": now,
                "indexed_at": now,
            }

            print(
                f"[{number}/{len(urls)}] indexed {action}: "
                f"{url} | lang={lang} | "
                f"doc_type={meta.get('doc_type', 'content')} | "
                f"chunks={len(chunks)}"
            )

            time.sleep(0.05)

        except Exception as e:
            page_errors += 1

            print(
                f"[{number}/{len(urls)}] "
                f"ERROR {url}: {e}"
            )

            # Preserve previous state for this URL on errors.
            if url in old_pages:
                new_pages[url] = old_pages[url]

    # -------------------------------------------------------------------------
    # Prune URLs that disappeared from the sitemap
    # -------------------------------------------------------------------------

    if PRUNE:
        previous_urls = set(old_pages.keys())
        removed_urls = sorted(
            previous_urls - sitemap_url_set
        )

        for url in removed_urls:
            try:
                print("prune removed sitemap URL:", url)

                delete_existing_points_for_url(
                    qd,
                    COLLECTION,
                    url,
                )

                new_pages.pop(url, None)
                pruned += 1

            except Exception as e:
                page_errors += 1
                print(
                    f"ERROR pruning {url}: {e}"
                )

    # Remove state entries that should not be in this site anymore
    # only when PRUNE is explicitly enabled.
    if PRUNE:
        for url in list(new_pages.keys()):
            if url not in sitemap_url_set:
                new_pages.pop(url, None)

    run_finished = utc_now()

    state = {
        "schema_version": 2,
        "site": SITE,
        "collection": COLLECTION,
        "sitemap_index": SITEMAP_INDEX,
        "embed_model": EMBED_MODEL,
        "last_run": run_finished,
        "last_run_status": (
            "ok"
            if page_errors == 0
            else "partial"
        ),
        "stats": {
            "urls_in_sitemap": len(urls),
            "indexed_new": indexed_new,
            "indexed_changed": indexed_changed,
            "unchanged": unchanged,
            "bootstrapped": bootstrapped,
            "skipped": skipped,
            "pruned": pruned,
            "errors": page_errors,
            "points_written": total_points_written,
        },
        "pages": new_pages,
    }

    save_state(state)

    print("")
    print("============================================================")
    print("DONE")
    print("============================================================")
    print("site:", SITE)
    print("collection:", COLLECTION)
    print("urls in sitemap:", len(urls))
    print("new pages indexed:", indexed_new)
    print("changed pages indexed:", indexed_changed)
    print("unchanged pages:", unchanged)
    print("baseline existing pages:", bootstrapped)
    print("skipped pages:", skipped)
    print("pruned pages:", pruned)
    print("page errors:", page_errors)
    print("points written:", total_points_written)
    print("state file:", state_file_path())
    print("last run:", run_finished)
    print(
        "status:",
        "ok" if page_errors == 0 else "partial",
    )
    print("============================================================")


if __name__ == "__main__":
    main()
