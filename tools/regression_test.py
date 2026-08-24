#!/usr/bin/env python3

import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests


ENDPOINT = "https://bot.coincharge.io/chat"
TIMEOUT = 45


@dataclass
class TestCase:
    name: str
    message: str
    site: str
    lang: str
    origin: str

    expected_collection: Optional[str] = None
    expected_guardrails: List[str] = field(
        default_factory=lambda: ["ok", "ok_repaired"]
    )

    require_backend: Optional[str] = "openai_direct"
    require_no_repair: bool = True
    require_suggestions: bool = True

    required_reply_terms: List[str] = field(default_factory=list)
    forbidden_reply_terms: List[str] = field(default_factory=list)
    required_reply_any_groups: List[List[str]] = field(default_factory=list)
    required_source_terms: List[str] = field(default_factory=list)

    required_meta_true: List[str] = field(default_factory=list)
    required_meta_false: List[str] = field(default_factory=list)
    required_meta_none: List[str] = field(default_factory=list)

    max_total_ms: Optional[int] = 10000


TESTS = [
    TestCase(
        name="Coinsnap Wallet Features",
        message=(
            "What features does the Coinsnap Bitcoin Point of Sale "
            "Wallet offer for merchants?"
        ),
        site="coinsnap.io",
        lang="en",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        required_reply_terms=[
            "self-custodial",
            "lightning",
            "bitcoin",
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
    ),

    TestCase(
        name="Coinsnap Wallet Self Custody",
        message=(
            "Is the Coinsnap POS Wallet self-custodial and who "
            "controls the private keys?"
        ),
        site="coinsnap.io",
        lang="en",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        required_reply_terms=[
            "self-custodial",
            "private",
            "keys",
        ],
    ),

    TestCase(
        name="Coinsnap Wallet Product Catalog",
        message=(
            "Can I create products and categories in the Coinsnap "
            "POS Wallet?"
        ),
        site="coinsnap.io",
        lang="en",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        required_reply_terms=[
            "product",
            "categor",
        ],
    ),

    TestCase(
        name="Coinsnap Wallet Analytics",
        message=(
            "What analytics and export features does the Coinsnap "
            "POS Wallet provide?"
        ),
        site="coinsnap.io",
        lang="en",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        required_reply_terms=[
            "analytics",
            "export",
        ],
    ),

    TestCase(
        name="Coinsnap WooCommerce",
        message=(
            "How can I accept Bitcoin payments with Coinsnap "
            "on WooCommerce?"
        ),
        site="coinsnap.io",
        lang="en",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        required_reply_terms=[
            "woocommerce",
            "coinsnap",
        ],
    ),

    TestCase(
        name="Developer Webhooks",
        message=(
            "How do Coinsnap webhooks work and how is the "
            "signature verified?"
        ),
        site="coincharge.io",
        lang="en",
        origin="https://coincharge.io",
        expected_collection="kb_coinsnap_docs_v2",
        required_reply_terms=[
            "webhook",
            "signature",
        ],
        required_source_terms=[
            "docs.coinsnap.io",
        ],
    ),

    TestCase(
        name="Developer API Keys",
        message=(
            "How do I create and use Coinsnap API keys for "
            "a custom integration?"
        ),
        site="docs.coinsnap.io",
        lang="en",
        origin="https://docs.coinsnap.io",
        expected_collection="kb_coinsnap_docs_v2",
        required_reply_terms=[
            "api",
            "key",
        ],
        required_source_terms=[
            "docs.coinsnap.io",
        ],
    ),

    TestCase(
        name="Developer Create Invoice",
        message=(
            "How do I create a Coinsnap invoice through the API?"
        ),
        site="docs.coinsnap.io",
        lang="en",
        origin="https://docs.coinsnap.io",
        expected_collection="kb_coinsnap_docs_v2",
        required_reply_terms=[
            "invoice",
            "api",
        ],
    ),

    TestCase(
        name="Developer Payment Links",
        message=(
            "How do Coinsnap payment links work and how can "
            "I create one?"
        ),
        site="docs.coinsnap.io",
        lang="en",
        origin="https://docs.coinsnap.io",
        expected_collection="kb_coinsnap_docs_v2",
        required_reply_terms=[
            "payment",
            "link",
        ],
    ),

    TestCase(
        name="Developer Bitcoin vs Lightning",
        message=(
            "What is the difference between Bitcoin on-chain "
            "and Lightning when integrating Coinsnap?"
        ),
        site="docs.coinsnap.io",
        lang="en",
        origin="https://docs.coinsnap.io",
        expected_collection="kb_coinsnap_docs_v2",
        required_reply_terms=[
            "bitcoin",
            "lightning",
        ],
    ),

    TestCase(
        name="Developer WordPress Integration",
        message=(
            "How can I integrate Coinsnap into a custom "
            "WordPress plugin?"
        ),
        site="docs.coinsnap.io",
        lang="en",
        origin="https://docs.coinsnap.io",
        expected_collection="kb_coinsnap_docs_v2",
        required_reply_terms=[
            "wordpress",
        ],
    ),

    TestCase(
        name="BTCPay Routing",
        message=(
            "Was ist ein BTCPay Server und wie kann ich ihn installieren?"
        ),
        site="coincharge.io",
        lang="de",
        origin="https://coincharge.io",
        expected_collection="kb_coincharge_v2",
        required_reply_terms=[
            "btcpay",
        ],
    ),

    TestCase(
        name="Scope Matrix Bringin Minimum",
        message=(
            "Wie hoch ist der Mindestbetrag für eine Zahlung "
            "über Bringin?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "11,000",
                "11.000",
            ],
            [
                "sat",
                "sats",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Bringin Maximum Payment",
        message=(
            "Wie hoch ist der Maximalbetrag pro Zahlung "
            "bei Bringin?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "3.000",
                "3,000",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Bringin Conversion Fee",
        message=(
            "Welche Conversion-Gebühr gilt bei Bringin "
            "für die Umwandlung in Euro?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "1 %",
                "1%",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Bringin Bank Payout Fee",
        message=(
            "Welche Gebühr verlangt Bringin für die Auszahlung "
            "auf mein Bankkonto?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "1 %",
                "1%",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Bringin KYC Documents",
        message=(
            "Welche KYC-Anforderungen gelten für ein "
            "Bringin-Konto?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
            "kyc",
        ],
        required_reply_any_groups=[
            [
                "ausweis",
                "id",
                "ident",
            ],
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Bringin Payout Account",
        message=(
            "Welche Anforderung gilt bei Bringin für das "
            "Bankkonto, auf das ausgezahlt wird?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "name",
                "kontoinhaber",
                "inhaber",
            ],
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Bringin SEPA Payout",
        message=(
            "Wie erfolgt die Auszahlung bei Bringin nach "
            "der Bestätigung?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
            "sepa",
        ],
        required_reply_any_groups=[
            [
                "instant",
                "sofort",
            ],
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Wallet vs Bringin KYC",
        message=(
            "Welche KYC-Anforderung gilt für die Coinsnap Wallet, "
            "und welche gilt bei Bringin?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap wallet",
            "bringin",
            "kyc",
        ],
        forbidden_reply_terms=[
            "coinsnap wallet benötigt einen gültigen ausweis",
            "coinsnap wallet benötigt einen ausweis",
            "coinsnap wallet kyc bei bringin",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix Wallet vs Bringin Payout Fee",
        message=(
            "Gilt die 1-Prozent-Bankauszahlungsgebühr von Bringin "
            "auch für die Coinsnap Wallet?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap wallet",
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "1 %",
                "1%",
                "1-prozent",
                "1 prozent",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt 1",
            "bei der coinsnap wallet gilt eine gebühr von 1",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Matrix DFX vs Bringin Limit",
        message=(
            "Hat DFX dasselbe monatliche Auszahlungslimit von "
            "10.000 Euro wie Bringin?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "dfx",
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "10.000",
                "10,000",
            ],
        ],
        forbidden_reply_terms=[
            "dfx hat ein monatliches auszahlungslimit von 10",
            "bei dfx gilt ein monatliches auszahlungslimit von 10",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    # =========================================================
    # Adversarial / paraphrase scope regression matrix
    # =========================================================

    TestCase(
        name="Adversarial Wallet 10k Limit",
        message=(
            "Hat die Coinsnap Wallet ein 10k Auszahlungslimit?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap wallet",
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "10.000",
                "10,000",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet hat ein 10",
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Coinsnap More Than 10k",
        message=(
            "Kann ich bei Coinsnap mehr als 10.000 Euro auszahlen?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap",
        ],
        required_reply_any_groups=[
            [
                "nicht angegeben",
                "nicht spezifiziert",
                "kein wert",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap hat ein auszahlungslimit von 10",
            "bei coinsnap gilt ein auszahlungslimit von 10",
            "coinsnap auszahlungslimit beträgt 10",
            "coinsnap kann bis zu 10.000",
            "coinsnap kann bis zu 10,000",
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Payout Fee Short",
        message=(
            "Was kostet die Auszahlung aufs Bankkonto?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_any_groups=[
            [
                "nicht angegeben",
                "nicht spezifiziert",
                "dfx",
                "bringin",
                "partner",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap auszahlungsgebühr beträgt",
            "coinsnap verlangt für die auszahlung",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Bringin Limit Short",
        message=(
            "Bringin Limit erhöhen wo?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "kyc",
                "dokument",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Coinsnap KYC Assumption",
        message=(
            "Muss ich bei der Coinsnap Wallet KYC machen, "
            "um mehr auszahlen zu können?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap wallet",
            "bringin",
            "kyc",
        ],
        forbidden_reply_terms=[
            "coinsnap wallet verlangt kyc",
            "coinsnap wallet erfordert kyc",
            "kyc bei der coinsnap wallet erforderlich",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial One Percent Coinsnap",
        message=(
            "Ist die 1% Auszahlungsgebühr von Coinsnap?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap",
        ],
        required_reply_any_groups=[
            [
                "nicht angegeben",
                "nicht spezifiziert",
                "kein wert",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap ist 1% auszahlungsgebühr",
            "coinsnap ist 1 % auszahlungsgebühr",
            "coinsnap verlangt 1%",
            "coinsnap verlangt 1 %",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Who Charges Payout Fee",
        message=(
            "Coinsnap oder Bringin – wer nimmt die "
            "Auszahlungsgebühr?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "1 %",
                "1%",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap nimmt die auszahlungsgebühr",
            "coinsnap verlangt die auszahlungsgebühr",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Foreign Payout Account",
        message=(
            "Kann ich bei Bringin auf das Bankkonto "
            "meiner Frau auszahlen?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "gleichen namen",
                "selben namen",
                "same name",
                "not possible",
                "nicht möglich",
                "nicht moeglich",
            ],
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial SEPA Instant Short",
        message=(
            "Gehen Bringin Auszahlungen sofort per SEPA?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
            "sepa",
        ],
        required_reply_any_groups=[
            [
                "instant",
                "sofort",
            ],
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial DFX vs Bringin Fee",
        message=(
            "Was kostet DFX und gilt das auch für Bringin?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "dfx",
            "bringin",
        ],
        forbidden_reply_terms=[
            "dfx und bringin verlangen dieselbe gebühr",
            "gleiche gebühr bei dfx und bringin",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Wallet Daily Limit Short",
        message=(
            "Wallet tägliches Limit?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "angegeben",
        ],
        forbidden_reply_terms=[
            "tägliches limit von 10.000",
            "tägliches limit von 10,000",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Bringin Monthly Limit",
        message=(
            "Wie viel kann ich mit Bringin im Monat auszahlen?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "10.000",
                "10,000",
            ],
            [
                "monat",
                "monthly",
            ],
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Bringin Three Thousand",
        message=(
            "Sind 3000 Euro das Coinsnap Zahlungslimit?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "3.000",
                "3,000",
                "3000",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap zahlungslimit beträgt",
            "coinsnap wallet beträgt 3",
            "coinsnap wallet gilt",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Bringin Minimum Casual",
        message=(
            "Wie klein darf eine Bringin Zahlung sein?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
        ],
        required_reply_any_groups=[
            [
                "11.000",
                "11,000",
            ],
            [
                "sat",
                "sats",
            ],
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Adversarial Bringin Fee Compound",
        message=(
            "Wer berechnet die Bankauszahlungsgebühr "
            "und wie hoch ist sie?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_any_groups=[
            [
                "nicht angegeben",
                "nicht spezifiziert",
                "dfx",
                "bringin",
                "partner",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap berechnet die bankauszahlungsgebühr",
            "coinsnap verlangt die bankauszahlungsgebühr",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Direct Bringin",
        message=(
            "Wie hoch ist das monatliche Auszahlungslimit bei "
            "Bringin und kann es durch zusätzliche "
            "KYC-Dokumente erhöht werden?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "bringin",
            "auszahlungslimit",
            "kyc",
        ],
        required_reply_any_groups=[
            [
                "10.000",
                "10,000",
            ],
        ],
        forbidden_reply_terms=[
            "coinsnap wallet beträgt",
            "coinsnap wallet gilt",
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Direct DFX",
        message=(
            "Welche Rolle hat DFX bei der Auszahlung von "
            "Bitcoin-Umsätzen in EUR auf ein Bankkonto?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "dfx",
            "eur",
        ],
        required_reply_any_groups=[
            [
                "bankkonto",
                "bank",
            ],
        ],
        forbidden_reply_terms=[
            "10.000",
            "10,000",
            "bringin-limit",
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Direct Coinsnap Customer Account",
        message=(
            "Wie hoch ist der Mindestbetrag zum Aufladen "
            "des Coinsnap Customer Accounts?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap customer account",
        ],
        required_reply_any_groups=[
            [
                "€5",
                "5 €",
                "5 euro",
            ],
        ],
        forbidden_reply_terms=[
            "bringin",
            "dfx",
            "10.000",
            "10,000",
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
        max_total_ms=12000,
    ),

    TestCase(
        name="Scope Ownership Bringin vs Coinsnap Wallet",
        message=(
            "Wie hoch ist das tägliche Auszahlungslimit "
            "der Coinsnap Wallet und kann ich dieses "
            "Limit im Dashboard erhöhen?"
        ),
        site="coinsnap.io",
        lang="de",
        origin="https://coinsnap.io",
        expected_collection="kb_coinsnap_v2",
        require_no_repair=False,
        require_suggestions=False,
        required_reply_terms=[
            "coinsnap wallet",
            "angegeben",
            "bringin",
            "dashboard",
        ],
        required_reply_any_groups=[
            [
                "10.000",
                "10,000",
            ],
        ],
        forbidden_reply_terms=[
            "tägliches auszahlungslimit von 10.000",
            "coinsnap wallet gilt",
            "coinsnap wallet beträgt",
        ],
        required_source_terms=[
            "coinsnap.io",
        ],
        required_meta_true=[
            "scope_structured_used",
            "repair_attempted",
            "repair_success",
            "scope_repair_success",
        ],
        required_meta_false=[
            "scope_audit_attempted",
        ],
        required_meta_none=[
            "scope_fact_extraction_error",
        ],
        max_total_ms=10000,
    ),

    TestCase(
        name="Coinpages Frankfurt",
        message=(
            "Wo kann ich in Frankfurt mit Bitcoin bezahlen?"
        ),
        site="coincharge.io",
        lang="de",
        origin="https://coincharge.io",
        expected_collection="kb_coinpages_v2",
        required_reply_terms=[
            "frankfurt",
        ],
        required_source_terms=[
            "coinpages.io",
        ],
    ),
]



# Optional targeted test execution.
#
# Example:
# python3 tools/regression_test.py --match "Scope Ownership"
_match_value = None

if "--match" in sys.argv:
    try:
        _match_index = sys.argv.index(
            "--match"
        )

        _match_value = sys.argv[
            _match_index + 1
        ].strip()

    except (
        ValueError,
        IndexError,
    ):
        print(
            "ERROR: --match requires a test name fragment"
        )
        raise SystemExit(2)

if _match_value:
    TESTS = [
        test
        for test in TESTS
        if _match_value.casefold()
        in test.name.casefold()
    ]

    if not TESTS:
        print(
            "ERROR: no regression tests matched: "
            + _match_value
        )
        raise SystemExit(2)



def contains_all(text: str, terms: List[str]):
    lower = (text or "").lower()

    missing = [
        term
        for term in terms
        if term.lower() not in lower
    ]

    return missing


def run_test(test: TestCase, index: int):
    payload = {
        "message": test.message,
        "sessionId": f"regression-{index}-{int(time.time())}",
        "site": test.site,
        "lang": test.lang,
    }

    started = time.perf_counter()

    try:
        response = requests.post(
            ENDPOINT,
            headers={
                "Content-Type": "application/json",
                "Origin": test.origin,
            },
            json=payload,
            timeout=TIMEOUT,
        )

        elapsed_ms = int(
            (time.perf_counter() - started)
            * 1000
        )

    except Exception as exc:
        return False, {
            "error": f"request failed: {exc}",
        }

    if response.status_code != 200:
        return False, {
            "error": f"HTTP {response.status_code}",
            "body": response.text[:500],
        }

    try:
        data = response.json()
    except Exception as exc:
        return False, {
            "error": f"invalid JSON: {exc}",
            "body": response.text[:500],
        }

    meta = data.get("meta") or {}
    reply = data.get("reply") or ""
    sources = data.get("sources") or []
    suggestions = data.get("suggestions") or []

    failures = []

    guardrail = meta.get("guardrail")

    if guardrail not in test.expected_guardrails:
        failures.append(
            f"guardrail={guardrail}"
        )

    if test.expected_collection:
        actual_collection = meta.get(
            "preferred_collection"
        )

        if actual_collection != test.expected_collection:
            failures.append(
                "preferred_collection="
                + str(actual_collection)
                + " expected="
                + test.expected_collection
            )

    if test.require_backend:
        backend = meta.get(
            "generation_backend"
        )

        if backend != test.require_backend:
            failures.append(
                "generation_backend="
                + str(backend)
                + " expected="
                + test.require_backend
            )

    if test.require_no_repair:
        if meta.get("repair_attempted") is True:
            failures.append(
                "repair_attempted=True"
            )

    if test.require_suggestions:
        if len(suggestions) != 3:
            failures.append(
                f"suggestions={len(suggestions)} expected=3"
            )

    missing_reply = contains_all(
        reply,
        test.required_reply_terms,
    )

    if missing_reply:
        failures.append(
            "reply missing: "
            + ", ".join(missing_reply)
        )

    reply_lower = reply.lower()

    for group in test.required_reply_any_groups:
        if not any(
            term.lower() in reply_lower
            for term in group
        ):
            failures.append(
                "reply missing one-of: "
                + " | ".join(group)
            )

    forbidden_found = [
        term
        for term in test.forbidden_reply_terms
        if term.lower() in reply_lower
    ]

    if forbidden_found:
        failures.append(
            "reply contains forbidden: "
            + ", ".join(forbidden_found)
        )

    for key in test.required_meta_true:
        if meta.get(key) is not True:
            failures.append(
                f"meta.{key}="
                + repr(meta.get(key))
                + " expected=True"
            )

    for key in test.required_meta_false:
        if meta.get(key) is not False:
            failures.append(
                f"meta.{key}="
                + repr(meta.get(key))
                + " expected=False"
            )

    for key in test.required_meta_none:
        if meta.get(key) is not None:
            failures.append(
                f"meta.{key}="
                + repr(meta.get(key))
                + " expected=None"
            )

    source_text = " ".join(
        (
            str(source.get("title", ""))
            + " "
            + str(source.get("url", ""))
        )
        for source in sources
    )

    missing_sources = contains_all(
        source_text,
        test.required_source_terms,
    )

    if missing_sources:
        failures.append(
            "sources missing: "
            + ", ".join(missing_sources)
        )

    total_ms = meta.get(
        "total_ms"
    )

    if (
        test.max_total_ms is not None
        and isinstance(total_ms, (int, float))
        and total_ms > test.max_total_ms
    ):
        failures.append(
            f"slow={total_ms}ms "
            f"limit={test.max_total_ms}ms"
        )

    result = {
        "http_ms": elapsed_ms,
        "total_ms": total_ms,
        "retrieval_ms": meta.get(
            "retrieval_ms"
        ),
        "guardrail": guardrail,
        "backend": meta.get(
            "generation_backend"
        ),
        "repair": meta.get(
            "repair_attempted"
        ),
        "preferred_collection": meta.get(
            "preferred_collection"
        ),
        "sources": len(sources),
        "suggestions": len(suggestions),
        "failures": failures,
    }

    return len(failures) == 0, result


def main():
    print()
    print("=" * 78)
    print("Coincharge / Coinsnap Support Regression Test")
    print("=" * 78)
    print()

    passed = 0
    failed = 0

    rows = []

    for index, test in enumerate(
        TESTS,
        start=1,
    ):
        print(
            f"[{index:02d}/{len(TESTS):02d}] "
            f"{test.name} ... ",
            end="",
            flush=True,
        )

        ok, result = run_test(
            test,
            index,
        )

        if ok:
            passed += 1
            status = "PASS"
            print("PASS")
        else:
            failed += 1
            status = "FAIL"
            print("FAIL")

        rows.append(
            (
                test,
                status,
                result,
            )
        )

    print()
    print("=" * 78)
    print("RESULTS")
    print("=" * 78)

    for test, status, result in rows:
        print()
        print(
            f"{status:<4}  {test.name}"
        )

        if "error" in result:
            print(
                "      ERROR:",
                result["error"],
            )
            continue

        print(
            "      total_ms:",
            result.get("total_ms"),
            "| retrieval_ms:",
            result.get("retrieval_ms"),
            "| backend:",
            result.get("backend"),
        )

        print(
            "      collection:",
            result.get(
                "preferred_collection"
            ),
            "| guardrail:",
            result.get("guardrail"),
            "| repair:",
            result.get("repair"),
        )

        if result.get("failures"):
            for failure in result["failures"]:
                print(
                    "      -",
                    failure,
                )

    print()
    print("=" * 78)
    print(
        f"PASS: {passed}   "
        f"FAIL: {failed}   "
        f"TOTAL: {len(TESTS)}"
    )
    print("=" * 78)
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
