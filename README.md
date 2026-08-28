# Coincharge Support Chat

**AI-powered Bitcoin, Lightning and Coinsnap support assistant with source-aware retrieval, evidence grounding and WordPress integration.**

Coincharge Support Chat is an AI-powered support and search system for Bitcoin, Lightning and Coinsnap-related content.

Instead of answering from general AI knowledge alone, the system retrieves relevant information from dedicated knowledge bases built from content published on:

- [coincharge.io](https://coincharge.io/) – Bitcoin and Lightning guides, payment solutions, BTCPay Server, merchant information and related content
- [coinsnap.io](https://coinsnap.io/) – Coinsnap products, Bitcoin payment solutions, plugins, merchant features and support content
- [coinpages.io](https://coinpages.io/) – Bitcoin acceptance locations and merchant information
- [docs.coinsnap.io](https://docs.coinsnap.io/) – Coinsnap developer documentation, API integration, invoices, webhooks and technical implementation guides

The assistant searches these sources, selects relevant passages and uses them as context for generating an answer.

The goal is to provide useful answers that are grounded in the underlying source material.

---

## WordPress Plugin

The easiest way to use Coincharge Support Chat on a WordPress website is with the **Coincharge Support Chat WordPress Plugin**.

### Download

**Latest version: v1.6.0**

[Download Coincharge Support Chat WordPress Plugin](https://github.com/onleines/coincharge-bot/releases/latest)

The installable WordPress ZIP file can be found under **Assets** on the release page.

Current plugin file:

```text
coincharge-support-chat-1.6.0.zip
```

---

## Quick Start

### 1. Install the WordPress plugin

Download the latest plugin ZIP from:

https://github.com/onleines/coincharge-bot/releases/latest

In WordPress:

1. Open **Plugins → Add New Plugin**
2. Select **Upload Plugin**
3. Upload `coincharge-support-chat-1.6.0.zip`
4. Install and activate the plugin
5. Open the plugin settings and verify the Support API URL

The default API endpoint is:

```text
https://bot.coincharge.io/chat
```

---

### 2. Create the Support Chat page

Each WordPress website using the plugin needs a dedicated support page.

Create a new WordPress page with the slug:

```text
bitcoin-support-chat
```

The resulting URL will normally be:

```text
https://example.com/bitcoin-support-chat/
```

Add the following shortcode to the page:

```text
[cc_support_search]
```

Publish the page.

The plugin uses this page to display the support interface, generated answers, sources and suggested follow-up questions.

---

### 3. Test the Support Chat

Open:

```text
https://example.com/bitcoin-support-chat/
```

and enter a question.

Example installations:

- https://leinert.com/bitcoin-support-chat/
- https://coincharge.io/bitcoin-support-chat/

---

## Using the plugin on multiple websites

The WordPress plugin is designed so that one central Coincharge Support Chat backend can serve multiple websites.

Each WordPress installation only needs the plugin and a connection to the central Support API.

For example:

```text
leinert.com
     │
coincharge.io
     │
coinpages.io
     │
coinsnap.io
     │
     ▼
WordPress Support Plugin
     │
     ▼
https://bot.coincharge.io/chat
     │
     ▼
Coincharge Support Backend
```

This makes it possible to maintain knowledge retrieval and AI logic centrally while providing the support interface on multiple WordPress websites.

The originating website and language can be passed to the backend so that retrieval and responses can be adapted to the context of the website.

---

## Global Support Search

In addition to the dedicated Support Chat page, the WordPress plugin can provide a global support search interface on the website.

Visitors can enter a question from another page and are then directed to:

```text
/bitcoin-support-chat/?q=...
```

The Support Chat page can automatically process the question and display the generated answer.

The global support search can be enabled or disabled in the WordPress plugin settings.

---

## How it works

A typical request follows this flow:

```text
Visitor
   │
   ▼
WordPress Support Plugin
   │
   ▼
Coincharge Support API
   │
   ▼
Query Routing & Retrieval
   │
   ▼
Knowledge Bases
   ├── coincharge.io
   ├── coinsnap.io
   ├── coinpages.io
   └── docs.coinsnap.io
   │
   ▼
Evidence-grounded AI answer
   │
   ▼
WordPress Support Chat
```

The backend determines which knowledge source is most relevant to the user's question.

For example:

- Questions about Bitcoin, Lightning or BTCPay Server can be routed primarily to Coincharge content.
- Questions about Coinsnap products and plugins can prioritize Coinsnap content.
- Questions such as "Where can I pay with Bitcoin in Frankfurt?" can use Coinpages.
- Developer questions about API keys, invoices or webhooks can prioritize the Coinsnap developer documentation.

The retrieval system can also combine information from multiple sources when appropriate.

---

## Knowledge Sources

Coincharge Support Chat currently uses separate knowledge collections for different content sources.

| Source | Purpose |
| --- | --- |
| [coincharge.io](https://coincharge.io/) | Bitcoin, Lightning, BTCPay Server, merchant and payment information |
| [coinsnap.io](https://coinsnap.io/) | Coinsnap products, plugins, merchant and support content |
| [coinpages.io](https://coinpages.io/) | Bitcoin acceptance locations and merchant directory content |
| [docs.coinsnap.io](https://docs.coinsnap.io/) | Coinsnap developer and API documentation |

The backend can route queries between these collections and combine results when necessary.

Developer queries, for example, can prioritize the Coinsnap developer documentation, while location-related queries can prioritize Coinpages.

---

## Evidence Grounding

A major goal of the project is to reduce unsupported or incorrectly attributed AI answers.

The system therefore uses retrieved source material as evidence before generating an answer.

Additional grounding logic is used for questions where similar facts may belong to different products or providers.

This is particularly important when content mentions multiple services, fees, limits, account requirements or payment providers in the same article.

The assistant should distinguish between:

```text
What does the source say?
```

and:

```text
Which product or provider does this information actually belong to?
```

When the available source material does not provide a sufficiently clear answer, the assistant should prefer saying that the information is not clearly specified rather than inventing an answer.

---

## Source-Aware Retrieval

The backend supports different retrieval paths depending on the question.

Current use cases include:

- Bitcoin and Lightning support
- Coinsnap product questions
- Coinsnap WordPress and e-commerce integrations
- BTCPay Server questions
- Bitcoin acceptance locations
- Coinsnap API questions
- Webhooks
- API keys
- Invoice creation
- Payment links
- Developer integrations

The system can retrieve information across the configured knowledge collections and prioritize a collection according to the detected intent.

---

## Architecture

The project consists of several components:

```text
WordPress
    │
    ▼
Coincharge Support Plugin
    │
    ▼
FastAPI Broker
    │
    ├── Query / Intent Routing
    ├── Hybrid Retrieval
    ├── Evidence Grounding
    ├── Answer Generation
    └── Question Analytics
             │
             ▼
           Qdrant
             │
             ├── Coincharge Knowledge Base
             ├── Coinsnap Knowledge Base
             ├── Coinpages Knowledge Base
             └── Coinsnap Developer Documentation
```

The backend uses Qdrant for the knowledge collections and OpenAI models for embeddings and answer generation.

---

## Updating the Knowledge Base

Website content is indexed separately from the WordPress plugin.

When articles on Coincharge, Coinsnap, Coinpages or the Coinsnap developer documentation are changed, the corresponding knowledge base can be re-indexed so that the support assistant uses the updated information.

This allows editors to improve the quality of AI answers by improving the underlying source content without modifying the WordPress frontend.

---

## Testing

The project includes regression tests for important routing and grounding scenarios.

Tests cover areas such as:

- Coinsnap Wallet
- WooCommerce
- Coinsnap developer documentation
- API keys
- Webhooks
- Invoice creation
- Payment links
- BTCPay Server
- Coinpages location queries
- Entity and ownership grounding

Run the regression suite with:

```bash
python3 tools/regression_test.py
```

---

## Repository Structure

A simplified overview:

```text
coincharge-bot/
├── broker/
│   └── Support API and retrieval logic
│
├── kb/
│   └── Knowledge-base ingestion
│
├── tools/
│   └── Regression and analysis tools
│
├── wordpress/
│   └── Coincharge Support Chat WordPress plugin
│
├── docs/
│   └── Project documentation
│
└── README.md
```

---

## Requirements

A complete self-hosted installation requires components such as:

- Docker / Docker Compose
- Python
- FastAPI
- Qdrant
- OpenAI API access
- A web server / reverse proxy
- WordPress for the optional WordPress frontend

Credentials and API keys must not be committed to the repository.

Use environment configuration for secrets.

---

## WordPress Plugin Features

The WordPress plugin can be used independently on multiple WordPress websites that connect to the same Coincharge Support backend.

The plugin provides:

- AI-powered Support Chat
- Support Chat via shortcode
- Global support search
- Central API connection
- Source display
- Suggested follow-up questions
- Multi-site usage
- Language-aware routing
- Optional local debug logging

### Shortcode

```text
[cc_support_search]
```

### Default Support Page

```text
/bitcoin-support-chat/
```

### Default API

```text
https://bot.coincharge.io/chat
```

### Latest Plugin Release

https://github.com/onleines/coincharge-bot/releases/latest

---

## Security

Do not commit:

```text
.env
API keys
OpenAI credentials
access tokens
passwords
Qdrant data
runtime data
logs containing sensitive information
```

Production secrets should always be supplied through environment variables or another secure configuration mechanism.

---

## Project Status

Coincharge Support Chat is an actively developed project.

Retrieval quality and answer grounding depend heavily on the quality and clarity of the indexed source content.

Both the software and the underlying content structure are therefore continuously improved.

---

## Maintainer

**Coincharge**

https://coincharge.io/

GitHub:

https://github.com/onleines/coincharge-bot
