import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DB_PATH = os.environ.get(
    "QUESTION_ANALYTICS_DB",
    "/app/data/question_analytics.sqlite3",
)

_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    directory = os.path.dirname(
        DB_PATH
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    connection = sqlite3.connect(
        DB_PATH,
        timeout=10,
    )

    connection.execute(
        "PRAGMA journal_mode=WAL"
    )

    connection.execute(
        "PRAGMA synchronous=NORMAL"
    )

    return connection


def init_db() -> None:
    with _lock:
        connection = _connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,

                    site TEXT NOT NULL,
                    lang TEXT NOT NULL,
                    question TEXT NOT NULL,

                    page_url TEXT,
                    page_path TEXT,

                    preferred_collection TEXT,
                    intent_reason TEXT,
                    developer_query INTEGER NOT NULL DEFAULT 0,
                    developer_intent_score REAL,

                    guardrail TEXT,
                    generation_backend TEXT,
                    answer_status TEXT,

                    retrieval_ms INTEGER,
                    total_ms INTEGER,

                    sources_count INTEGER NOT NULL DEFAULT 0,
                    source_urls_json TEXT,

                    repair_attempted INTEGER NOT NULL DEFAULT 0,
                    repair_success INTEGER,

                    is_test INTEGER NOT NULL DEFAULT 0,
                    has_context INTEGER NOT NULL DEFAULT 1
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_questions_created_at
                ON questions(created_at)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_questions_site
                ON questions(site)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_questions_collection
                ON questions(preferred_collection)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_questions_guardrail
                ON questions(guardrail)
                """
            )

            # Migration for existing installations.
            existing_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(questions)"
                ).fetchall()
            }

            if "answer_status" not in existing_columns:
                connection.execute(
                    """
                    ALTER TABLE questions
                    ADD COLUMN answer_status TEXT
                    """
                )

            if "is_test" not in existing_columns:
                connection.execute(
                    """
                    ALTER TABLE questions
                    ADD COLUMN is_test INTEGER NOT NULL DEFAULT 0
                    """
                )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_questions_answer_status
                ON questions(answer_status)
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                idx_questions_is_test
                ON questions(is_test)
                """
            )

            connection.commit()

        finally:
            connection.close()


def log_question(
    *,
    question: str,
    site: str,
    lang: str,
    page_url: Optional[str] = None,
    page_path: Optional[str] = None,
    retrieval_meta: Optional[Dict[str, Any]] = None,
    generation_meta: Optional[Dict[str, Any]] = None,
    sources: Optional[List[Dict[str, Any]]] = None,
    guardrail: Optional[str] = None,
    generation_backend: Optional[str] = None,
    answer_status: Optional[str] = None,
    total_ms: Optional[int] = None,
    is_test: bool = False,
    has_context: bool = True,
) -> None:

    question = (
        question
        or ""
    ).strip()

    if not question:
        return

    retrieval_meta = (
        retrieval_meta
        or {}
    )

    generation_meta = (
        generation_meta
        or {}
    )

    sources = (
        sources
        or []
    )

    source_urls = []

    for source in sources:
        url = str(
            source.get(
                "url",
                "",
            )
            or ""
        ).strip()

        if url:
            source_urls.append(
                url
            )

    # Keep URL order but remove duplicates.
    source_urls = list(
        dict.fromkeys(
            source_urls
        )
    )

    repair_success = (
        generation_meta.get(
            "repair_success"
        )
    )

    if repair_success is None:
        repair_success_db = None
    else:
        repair_success_db = (
            1
            if bool(repair_success)
            else 0
        )

    created_at = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    row = (
        created_at,
        site or "",
        lang or "",
        question,
        page_url,
        page_path,

        retrieval_meta.get(
            "preferred_collection"
        ),
        retrieval_meta.get(
            "intent_reason"
        ),
        (
            1
            if bool(
                retrieval_meta.get(
                    "developer_query"
                )
            )
            else 0
        ),
        retrieval_meta.get(
            "developer_intent_score"
        ),

        guardrail,
        generation_backend,
        answer_status,

        retrieval_meta.get(
            "retrieval_ms"
        ),
        total_ms,

        len(
            source_urls
        ),
        json.dumps(
            source_urls,
            ensure_ascii=False,
        ),

        (
            1
            if bool(
                generation_meta.get(
                    "repair_attempted"
                )
            )
            else 0
        ),
        repair_success_db,

        1 if is_test else 0,
        1 if has_context else 0,
    )

    try:
        with _lock:
            connection = _connect()

            try:
                connection.execute(
                    """
                    INSERT INTO questions (
                        created_at,
                        site,
                        lang,
                        question,
                        page_url,
                        page_path,
                        preferred_collection,
                        intent_reason,
                        developer_query,
                        developer_intent_score,
                        guardrail,
                        generation_backend,
                        answer_status,
                        retrieval_ms,
                        total_ms,
                        sources_count,
                        source_urls_json,
                        repair_attempted,
                        repair_success,
                        is_test,
                        has_context
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    row,
                )

                connection.commit()

            finally:
                connection.close()

    except Exception as exc:
        # Analytics must never break the support chat.
        print(
            "[QUESTION_ANALYTICS_ERROR] "
            + str(exc)[:300],
            flush=True,
        )


def get_summary(
    limit: int = 20,
) -> List[Dict[str, Any]]:
    connection = _connect()

    try:
        cursor = connection.execute(
            """
            SELECT
                id,
                created_at,
                site,
                lang,
                question,
                preferred_collection,
                intent_reason,
                developer_query,
                guardrail,
                generation_backend,
                answer_status,
                retrieval_ms,
                total_ms,
                sources_count,
                is_test,
                has_context
            FROM questions
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                int(limit),
            ),
        )

        columns = [
            item[0]
            for item in cursor.description
        ]

        return [
            dict(
                zip(
                    columns,
                    row,
                )
            )
            for row in cursor.fetchall()
        ]

    finally:
        connection.close()


init_db()
