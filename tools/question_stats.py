#!/usr/bin/env python3

import argparse
import sqlite3
from collections import Counter
from statistics import mean, median
from datetime import datetime, timedelta, timezone


DEFAULT_DB = "/opt/coincharge-bot/broker/data/question_analytics.sqlite3"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Question analytics report for Coincharge/Coinsnap support chat"
    )

    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite database path (default: {DEFAULT_DB})",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback window in days (default: 30)",
    )

    parser.add_argument(
        "--site",
        default=None,
        help="Optional exact site filter, e.g. coinsnap.io or docs.coinsnap.io",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent content-gap questions to show (default: 20)",
    )

    return parser.parse_args()


def fmt_ms(value):
    if value is None:
        return "-"

    return f"{int(round(value)):,} ms".replace(",", ".")


def pct(part, total):
    if not total:
        return "0.0 %"

    return f"{(part / total) * 100:.1f} %"


def main():
    args = parse_args()

    since = (
        datetime.now(timezone.utc)
        - timedelta(days=args.days)
    ).isoformat()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row

    where = [
        "is_test = 0",
        "created_at >= ?",
    ]

    params = [
        since,
    ]

    if args.site:
        where.append(
            "site = ?"
        )
        params.append(
            args.site
        )

    where_sql = " AND ".join(where)

    rows = con.execute(
        f"""
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
            has_context
        FROM questions
        WHERE {where_sql}
        ORDER BY id DESC
        """,
        params,
    ).fetchall()

    total = len(rows)

    statuses = Counter(
        (row["answer_status"] or "unknown")
        for row in rows
    )

    sites = Counter(
        (row["site"] or "unknown")
        for row in rows
    )

    collections = Counter(
        (row["preferred_collection"] or "unknown")
        for row in rows
    )

    langs = Counter(
        (row["lang"] or "unknown")
        for row in rows
    )

    backends = Counter(
        (row["generation_backend"] or "unknown")
        for row in rows
    )

    total_times = [
        int(row["total_ms"])
        for row in rows
        if row["total_ms"] is not None
    ]

    retrieval_times = [
        int(row["retrieval_ms"])
        for row in rows
        if row["retrieval_ms"] is not None
    ]

    gap_rows = [
        row
        for row in rows
        if (row["answer_status"] or "") in {
            "unsupported",
            "partial",
            "no_context",
        }
    ]

    print()
    print("=" * 78)
    print("Question Analytics Report")
    print("=" * 78)

    print(
        f"Window:               last {args.days} days"
    )

    print(
        f"Site filter:          {args.site or 'all'}"
    )

    print(
        f"Real user questions:  {total}"
    )

    print()

    print("ANSWER STATUS")
    print("-" * 78)

    for status in [
        "answered",
        "partial",
        "unsupported",
        "no_context",
        "unknown",
    ]:
        count = statuses.get(status, 0)

        print(
            f"{status:<18} "
            f"{count:>6}   "
            f"{pct(count, total):>8}"
        )

    print()
    print("SITES")
    print("-" * 78)

    if sites:
        for name, count in sites.most_common():
            print(
                f"{name:<30} "
                f"{count:>6}   "
                f"{pct(count, total):>8}"
            )
    else:
        print("No data")

    print()
    print("LANGUAGES")
    print("-" * 78)

    if langs:
        for name, count in langs.most_common():
            print(
                f"{name:<30} "
                f"{count:>6}   "
                f"{pct(count, total):>8}"
            )
    else:
        print("No data")

    print()
    print("PRIMARY COLLECTIONS")
    print("-" * 78)

    if collections:
        for name, count in collections.most_common():
            print(
                f"{name:<30} "
                f"{count:>6}   "
                f"{pct(count, total):>8}"
            )
    else:
        print("No data")

    print()
    print("GENERATION BACKENDS")
    print("-" * 78)

    if backends:
        for name, count in backends.most_common():
            print(
                f"{name:<30} "
                f"{count:>6}   "
                f"{pct(count, total):>8}"
            )
    else:
        print("No data")

    print()
    print("PERFORMANCE")
    print("-" * 78)

    if total_times:
        print(
            f"Total median:         {fmt_ms(median(total_times))}"
        )

        print(
            f"Total average:        {fmt_ms(mean(total_times))}"
        )

        print(
            f"Total max:            {fmt_ms(max(total_times))}"
        )
    else:
        print("No total_ms data")

    if retrieval_times:
        print(
            f"Retrieval median:     {fmt_ms(median(retrieval_times))}"
        )

        print(
            f"Retrieval average:    {fmt_ms(mean(retrieval_times))}"
        )

    print()
    print("CONTENT GAP SUMMARY")
    print("-" * 78)

    print(
        f"Gap questions:        {len(gap_rows)}"
    )

    print(
        f"Gap rate:             {pct(len(gap_rows), total)}"
    )

    print()
    print(
        f"RECENT CONTENT-GAP QUESTIONS "
        f"(max {args.limit})"
    )
    print("-" * 78)

    if not gap_rows:
        print("No unsupported / partial / no_context questions found.")
    else:
        for row in gap_rows[:args.limit]:
            print()
            print(
                f"[{row['answer_status']}] "
                f"{row['created_at']}"
            )

            print(
                f"site:       {row['site']}"
            )

            print(
                f"lang:       {row['lang']}"
            )

            print(
                f"collection: {row['preferred_collection']}"
            )

            print(
                f"question:   {row['question']}"
            )

    print()
    print("=" * 78)

    con.close()


if __name__ == "__main__":
    main()
