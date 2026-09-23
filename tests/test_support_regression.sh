#!/usr/bin/env bash

set -u

API="https://bot.coincharge.io/chat"
FAILURES=0

pass() {
    echo "PASS: $1"
}

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

request() {
    local message="$1"
    local session="$2"

    curl -s "$API" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n \
            --arg message "$message" \
            --arg session "$session" \
            '{
                message: $message,
                site: "coinsnap.io",
                sessionId: $session
            }'
        )"
}

echo "============================================================"
echo "1. DFX multi-fact"
echo "============================================================"

R=$(request \
    "What fees, payout limits and KYC requirements apply when using DFX bank payouts with Coinsnap?" \
    "regression-dfx-multi"
)

echo "$R" | jq '{
    reply,
    answer_status: .meta.answer_status,
    guardrail: .meta.guardrail,
    scope_structured_used: .meta.scope_structured_used,
    scope_compound_structured_used: .meta.scope_compound_structured_used,
    scope_atomic_fact_count: .meta.scope_atomic_fact_count,
    scope_audit_attempted: .meta.scope_audit_attempted
}'

if echo "$R" | jq -e '
    .meta.answer_status == "answered"
    and .meta.guardrail == "ok_repaired"
    and .meta.scope_structured_used == true
    and .meta.scope_compound_structured_used == true
    and .meta.scope_atomic_fact_count == 3
    and .meta.scope_audit_attempted == false
    and (.reply | contains("bank payout fee: 0.2%"))
    and (.reply | contains("transaction fee: 1%"))
    and (.reply | contains("conversion spread: 1%"))

    # The prepaid Coinsnap credit qualifier must belong to the
    # Coinsnap transaction fee, never to the DFX payout fee.
    and (
        .reply
        | test(
            "For Coinsnap, transaction fee: 1%\\.[\\s\\S]{0,140}prepaid Coinsnap credit";
            "i"
        )
    )
    and (
        (
            .reply
            | test(
                "For DFX, bank payout fee: 0\\.2%\\.\\s*\\n\\s*\\n\\s*Additionally:[^\\n]*prepaid Coinsnap credit";
                "i"
            )
        )
        | not
    )

    and (.reply | contains("CHF 1,000"))
    and (.reply | contains("rolling 30-day"))
    and (.reply | contains("KYC verification is mandatory"))
' >/dev/null; then
    pass "DFX multi-fact"
else
    fail "DFX multi-fact"
fi


echo
echo "============================================================"
echo "2. WooCommerce multi-fact"
echo "============================================================"

R=$(request \
    "What fees and KYC requirements apply when I use Coinsnap for WooCommerce?" \
    "regression-woocommerce-multi"
)

echo "$R" | jq '{
    reply,
    answer_status: .meta.answer_status,
    guardrail: .meta.guardrail,
    scope_structured_used: .meta.scope_structured_used,
    scope_compound_structured_used: .meta.scope_compound_structured_used,
    scope_atomic_fact_count: .meta.scope_atomic_fact_count,
    scope_audit_attempted: .meta.scope_audit_attempted
}'

if echo "$R" | jq -e '
    .meta.answer_status == "answered"
    and .meta.guardrail == "ok_repaired"
    and .meta.scope_structured_used == true
    and .meta.scope_compound_structured_used == true
    and .meta.scope_atomic_fact_count == 2
    and .meta.scope_audit_attempted == false
    and (.reply | contains("1% per successful payment"))
    and (.reply | contains("No monthly fee"))
    and (.reply | contains("No Coinsnap KYC for direct wallet settlement"))
    and (.reply | contains("External fiat or off-ramp partners may require KYC or KYB"))
' >/dev/null; then
    pass "WooCommerce multi-fact"
else
    fail "WooCommerce multi-fact"
fi


echo
echo "============================================================"
echo "3. DFX single-fact"
echo "============================================================"

R=$(request \
    "What KYC requirements apply when using DFX bank payouts with Coinsnap?" \
    "regression-dfx-single"
)

echo "$R" | jq '{
    reply,
    answer_status: .meta.answer_status,
    guardrail: .meta.guardrail,
    scope_structured_used: .meta.scope_structured_used,
    scope_audit_attempted: .meta.scope_audit_attempted
}'

if echo "$R" | jq -e '
    .meta.answer_status == "answered"
    and .meta.guardrail == "ok_repaired"
    and .meta.scope_structured_used == true
    and .meta.scope_audit_attempted == false
    and (.reply | contains("DFX"))
    and (.reply | contains("KYC"))
    and (.reply | contains("EUR/CHF 1,000"))
' >/dev/null; then
    pass "DFX single-fact"
else
    fail "DFX single-fact"
fi


echo
echo "============================================================"
echo "4. Frankfurt unsupported / Coinsnap isolation"
echo "============================================================"

R=$(request \
    "Where can I pay with Bitcoin in Frankfurt?" \
    "regression-frankfurt-unsupported"
)

echo "$R" | jq '{
    reply,
    answer_status: .meta.answer_status,
    guardrail: .meta.guardrail,
    collections: .meta.collections,
    sources
}'

if echo "$R" | jq -e '
    .meta.answer_status == "unsupported"
    and .meta.guardrail == "ok"
    and (.sources | length) == 0
    and (.meta.collections | length) == 2
    and (.meta.collections | contains(["kb_coinsnap_v2"]))
    and (.meta.collections | contains(["kb_coinsnap_docs_v2"]))
' >/dev/null; then
    pass "Frankfurt unsupported / Coinsnap isolation"
else
    fail "Frankfurt unsupported / Coinsnap isolation"
fi


echo
echo "============================================================"

if [ "$FAILURES" -eq 0 ]; then
    echo "ALL REGRESSION TESTS PASSED"
    exit 0
fi

echo "$FAILURES REGRESSION TEST(S) FAILED"
exit 1
