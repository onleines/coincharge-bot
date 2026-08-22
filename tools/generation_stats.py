#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from statistics import mean, median


MARKER = "[GENERATION_METRIC] "


def percentile(values, percentile_value):
    if not values:
        return None

    ordered = sorted(values)

    rank = math.ceil(
        (percentile_value / 100.0)
        * len(ordered)
    )

    rank = max(
        1,
        min(
            rank,
            len(ordered),
        ),
    )

    return ordered[
        rank - 1
    ]


def load_events(since):
    cmd = [
        "docker",
        "compose",
        "logs",
        "--no-color",
        "--since",
        since,
        "broker",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print(
            "ERROR: docker compose logs failed",
            file=sys.stderr,
        )

        if result.stderr:
            print(
                result.stderr.strip(),
                file=sys.stderr,
            )

        sys.exit(
            result.returncode
        )

    events = []

    for line in result.stdout.splitlines():

        if MARKER not in line:
            continue

        raw = line.split(
            MARKER,
            1,
        )[1].strip()

        try:
            event = json.loads(
                raw
            )
        except json.JSONDecodeError:
            continue

        if event.get("event") != "generation":
            continue

        events.append(
            event
        )

    return events


def format_ms(value):
    if value is None:
        return "-"

    return f"{int(value):,} ms".replace(
        ",",
        ".",
    )


def format_percent(value):
    return f"{value:.2f} %"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Statistics for Coincharge "
            "GENERATION_METRIC logs"
        )
    )

    parser.add_argument(
        "--since",
        default="24h",
        help=(
            "Docker log time window, "
            "for example 24h, 12h, 7d. "
            "Default: 24h"
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output statistics as JSON",
    )

    args = parser.parse_args()

    events = load_events(
        args.since
    )

    if not events:

        print(
            f"No GENERATION_METRIC events found "
            f"for --since {args.since}"
        )
        return

    durations = [
        int(
            event.get(
                "duration_ms",
                0,
            )
            or 0
        )
        for event in events
        if event.get(
            "duration_ms"
        ) is not None
    ]

    backends = Counter(
        str(
            event.get(
                "backend",
                "unknown",
            )
        )
        for event in events
    )

    models = Counter(
        str(
            event.get(
                "model",
                "unknown",
            )
        )
        for event in events
    )

    successful = sum(
        1
        for event in events
        if event.get(
            "success"
        ) is True
    )

    failed = len(
        events
    ) - successful

    slow = sum(
        1
        for event in events
        if (
            event.get(
                "slow"
            ) is True
            or int(
                event.get(
                    "duration_ms",
                    0,
                )
                or 0
            ) >= 5000
        )
    )

    direct = backends.get(
        "openai_direct",
        0,
    )

    fallback = backends.get(
        "openclaw_fallback",
        0,
    )

    generation_failed = backends.get(
        "generation_failed",
        0,
    )

    fallback_rate = (
        fallback
        / len(events)
        * 100.0
    )

    slow_rate = (
        slow
        / len(events)
        * 100.0
    )

    failure_rate = (
        failed
        / len(events)
        * 100.0
    )

    system_prompt_chars = [
        int(
            event.get(
                "system_prompt_chars",
                0,
            )
            or 0
        )
        for event in events
    ]

    stats = {
        "window": args.since,
        "generations": len(events),
        "successful": successful,
        "failed": failed,
        "failure_rate_percent": round(
            failure_rate,
            2,
        ),
        "openai_direct": direct,
        "openclaw_fallback": fallback,
        "generation_failed": (
            generation_failed
        ),
        "fallback_rate_percent": round(
            fallback_rate,
            2,
        ),
        "slow_over_5s": slow,
        "slow_rate_percent": round(
            slow_rate,
            2,
        ),
        "duration_ms": {
            "min": min(
                durations
            ),
            "median": median(
                durations
            ),
            "mean": round(
                mean(
                    durations
                )
            ),
            "p95": percentile(
                durations,
                95,
            ),
            "max": max(
                durations
            ),
        },
        "system_prompt_chars": {
            "median": median(
                system_prompt_chars
            ),
            "max": max(
                system_prompt_chars
            ),
        },
        "backends": dict(
            backends
        ),
        "models": dict(
            models
        ),
    }

    if args.json:

        print(
            json.dumps(
                stats,
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    print()
    print("=" * 64)
    print("Coincharge Generation Statistics")
    print("=" * 64)

    print(
        f"Window:               {args.since}"
    )
    print(
        f"Generations:          {len(events)}"
    )
    print(
        f"Successful:           {successful}"
    )
    print(
        f"Failed:               {failed} "
        f"({format_percent(failure_rate)})"
    )

    print()

    print(
        f"OpenAI direct:        {direct}"
    )
    print(
        f"OpenClaw fallback:    {fallback}"
    )
    print(
        f"Fallback rate:        "
        f"{format_percent(fallback_rate)}"
    )

    print()

    print(
        f"Slow >= 5s:           {slow} "
        f"({format_percent(slow_rate)})"
    )

    print()

    print(
        f"Min:                  "
        f"{format_ms(min(durations))}"
    )
    print(
        f"Median:               "
        f"{format_ms(median(durations))}"
    )
    print(
        f"Average:              "
        f"{format_ms(round(mean(durations)))}"
    )
    print(
        f"P95:                  "
        f"{format_ms(percentile(durations, 95))}"
    )
    print(
        f"Max:                  "
        f"{format_ms(max(durations))}"
    )

    print()

    print(
        f"Median prompt chars:  "
        f"{int(median(system_prompt_chars)):,}"
        .replace(",", ".")
    )

    print(
        f"Max prompt chars:     "
        f"{max(system_prompt_chars):,}"
        .replace(",", ".")
    )

    print()
    print("Backends:")

    for backend, count in (
        backends.most_common()
    ):
        print(
            f"  {backend:<22} {count}"
        )

    print()
    print("Models:")

    for model, count in (
        models.most_common()
    ):
        print(
            f"  {model:<22} {count}"
        )

    print()
    print("=" * 64)


if __name__ == "__main__":
    main()
