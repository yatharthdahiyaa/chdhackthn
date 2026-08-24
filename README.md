# FusionWatch — Unified Analytics Platform MVP

> **Prototype / Demo / Research use only.**
> All data is fully synthetic. No real persons, phone numbers, IMEIs, bank accounts, or social handles are used anywhere in this codebase.

---

## What it does

FusionWatch fuses four synthetic data sources — **CDR** (Call Detail Records), **IPDR** (Internet Protocol Detail Records), **financial transactions**, and **social/OSINT posts** — into a single entity graph, runs anomaly detection across all sources, and surfaces results in an interactive dashboard.

```
python generate_synthetic_data.py && streamlit run app.py
```

*(Run the full pipeline first — see [Quick Start](#quick-start) below.)*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Ingestion                                             │
│  CDR.csv │ IPDR.csv │ transactions.csv │ social.json            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ ingest.py
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 2 — Normalization                                         │
│  Entity(entity_id, name, type)                                   │
│  Event(event_id, entity_id, source, timestamp_utc, attributes)  │
└──────────────────────┬──────────────────────────────────────────┘
                       │ entity_resolution.py
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 3 — Entity Resolution                                     │
│  Union-Find (shared IMEI/phone/account) + rapidfuzz name match  │
│  → NetworkX MultiDiGraph                                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │ analytics.py
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 4 — Analytics                                             │
│  Per-source: IsolationForest + rule-based flags                  │
│  Fused: Louvain community + PageRank + 4 cross-domain rules      │
│  → risk_scores.parquet (score + plain-English explanation)       │
└──────────────────────┬──────────────────────────────────────────┘
                       │ app.py
┌──────────────────────▼──────────────────────────────────────────┐
│  Layer 5 — Dashboard (Streamlit)                                 │
│  Overview │ Entity Profile │ Graph Explorer │ Geo Map            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- ~500 MB disk (for dependencies + data)

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the full pipeline

```bash
# Generate all four synthetic datasets
python generate_synthetic_data.py

# Ingest + normalize
python ingest.py

# Entity resolution (builds the graph)
python entity_resolution.py

# Analytics (anomaly detection + risk scoring)
python analytics.py

# Launch the dashboard
streamlit run app.py
```

Or as a single one-liner (Windows PowerShell):

```powershell
python generate_synthetic_data.py; python ingest.py; python entity_resolution.py; python analytics.py; streamlit run app.py
```

### 3. Run unit tests

```bash
pytest tests/ -v
```

---

## Deployment Options

### Option A: Streamlit Community Cloud (Recommended & Free)
1. Fork or push this repository to your GitHub account: `https://github.com/yatharthdahiyaa/chdhackthn.git`
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"** and select:
   - **Repository:** `yatharthdahiyaa/chdhackthn`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!** The app will auto-initialize synthetic datasets and launch immediately.

### Option B: Docker Container
Build and run anywhere with Docker:
```bash
# Build the Docker image
docker build -t fusionwatch:latest .

# Run the container on port 8501
docker run -p 8501:8501 fusionwatch:latest
```
Open `http://localhost:8501` in your browser.

### Option C: Render / PaaS Cloud Web Service
1. Connect your GitHub repository to [Render](https://render.com/).
2. Create a new **Web Service**:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
3. Click **Create Web Service**.

---

## Planted Anomalies (what to find in the demo)

The generator deliberately plants four types of anomalies for the demo to discover:

| Anomaly | Who | What |
|---------|-----|------|
| **Shared IMEI** | E000 + E001 | Two identities using the same physical device |
| **Circular transaction flow** | E000 → E001 → E002 → E000 | Classic transaction layering pattern |
| **Impossible travel** | E003 | Appears at Chandigarh and Delhi towers 25 min apart (≈250 km) |
| **Tor + large transfer** | E004 | Tor sessions concurrent with outbound transfers >₹75,000 |
| **Social burst** | All 5 suspects | 30 posts each within a 2-hour coordinated window |

---

## Cross-Domain Correlation Rules

| Rule | Trigger |
|------|---------|
| **CR-1** | CDR tower within 500 m of ATM + transaction within ±15 min |
| **CR-2** | Tor/anonymizer IPDR session + outbound transfer > ₹50,000 within 1 hour |
| **CR-3** | Social post burst (Z > 2.5) + transaction cluster within 24 hours |
| **CR-4** | Two CDR towers > 200 km apart in < 60 minutes (impossible travel) |

---

## Composite Risk Score

```
risk_score = 0.25 × CDR_score
           + 0.20 × IPDR_score
           + 0.30 × TX_score
           + 0.10 × Social_score
           + 0.15 × Graph_score
           + rule_boost (up to +0.20 for triggered cross-domain rules)
```

Each component is an IsolationForest outlier score or a rule-based flag, normalized to [0, 1]. The composite is a **weighted, explainable** score — not a black box. Every flagged entity includes a plain-English explanation of which signals contributed.

---

## Data Schemas

### CDR (`data/cdr.csv`)
| Column | Type | Description |
|--------|------|-------------|
| cdr_id | str | Unique record ID |
| timestamp | ISO 8601 +05:30 | Call/SMS time |
| calling_number | str | Synthetic E.164 (+91…) |
| called_number | str | Synthetic E.164 (+91…) |
| duration_sec | int | 0 for SMS/MMS |
| call_type | VOICE/SMS/MMS | — |
| roaming_flag | bool | — |
| tower_id | str | Cell tower identifier |
| tower_lat / tower_lon | float | Tower location (Punjab/Chandigarh bbox) |
| imei | str | 15-digit synthetic IMEI (prefix 99) |
| imsi | str | 15-digit synthetic IMSI (MCC=404, MNC=99) |
| entity_id | str | Synthetic entity reference |

### IPDR (`data/ipdr.csv`)
| Column | Type | Description |
|--------|------|-------------|
| ipdr_id | str | Unique record ID |
| timestamp | ISO 8601 | Session start |
| entity_id, imei, imsi | str | Device identifiers |
| src_ip / dst_ip | str | RFC 5737 documentation ranges (fictional) |
| src_port / dst_port | int | — |
| protocol | TCP/UDP/QUIC | — |
| session_duration_sec | int | — |
| bytes_up / bytes_down | int | Volume only — no payload |
| service_category | str | WhatsApp / YouTube / Tor / etc. |
| app_fingerprint | str | Machine-readable version of service_category |

### Transactions (`data/transactions.csv`)
| Column | Type | Description |
|--------|------|-------------|
| tx_id | str | Unique record ID |
| timestamp | ISO 8601 | Transaction time |
| entity_id / account_id | str | Sender |
| counterparty_account / counterparty_entity_id | str | Receiver |
| amount_inr | float | Amount in Indian Rupees |
| tx_type | NEFT/IMPS/UPI/Cash/RTGS | — |
| channel | Mobile-App / ATM / Branch / etc. | — |
| reference_note | str | Synthetic free-text memo |

### Social/OSINT (`data/social.json`)
```json
{
  "post_id": "POST_E000_012345",
  "entity_id": "E000",
  "handle": "synthetic_handle42",
  "timestamp": "2025-03-15T14:22:00+05:30",
  "text": "Just arrived at Chandigarh. Great weather!",
  "mentions": ["@other_handle99"],
  "hashtags": ["fintech"],
  "geotag": {"lat": 30.733, "lon": 76.779}
}
```

---

## Legal / Ethical Framing (India — Design Assumptions)

This prototype is built for **academic/research/hackathon demonstration only**. The following regulatory frameworks apply to real production deployments — they are cited here as design assumptions, not implemented enforcement:

| Domain | Relevant Framework |
|--------|-------------------|
| CDR/IPDR collection | Indian Telegraph Act 1885, §5(2); IT Act 2000; DoT TRAI regulations |
| Financial transaction monitoring | RBI Master Directions on KYC; Prevention of Money Laundering Act (PMLA) 2002; FATF Recommendations |
| OSINT / social media analysis | IT Act 2000, §43 & §66; Digital Personal Data Protection Act (DPDPA) 2023 |
| Data retention and audit | DPDPA fiduciary obligations; RBI record retention norms |

**Key framing**: All entity names, phone numbers, IMEIs, IMSIs, account numbers, IP addresses, and social handles in this repository are randomly generated and do not correspond to any real person, device, account, or network.

---

## Production Considerations (Out of Scope for MVP)

The following would be required before any production or compliance deployment:

- **Authentication & authorization**: multi-user case management, RBAC
- **Chain-of-custody / audit logging**: every query and action must be logged with investigator ID and timestamp
- **DPDPA-grade data protection**: data minimization, purpose limitation, consent management, erasure workflows
- **Real-time streaming**: Kafka/Flink for live CDR/IPDR feeds
- **Scalable storage**: Hadoop/Spark for TB-scale CDR; Neo4j or Amazon Neptune for production graph DB
- **Explainability certification**: model cards, bias audits for any ML components used in enforcement decisions
- **Legal process integration**: warrant management, court-admissible export formats

---

## File Structure

```
chandigarh-policr/
├── config.py                   # Central configuration
├── generate_synthetic_data.py  # Synthetic data generator (all 4 sources)
├── ingest.py                   # Ingestion + normalization
├── entity_resolution.py        # Entity graph builder
├── analytics.py                # Anomaly detection + risk scoring
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── data/                       # Generated CSV/JSON (git-ignored)
│   ├── cdr.csv
│   ├── ipdr.csv
│   ├── transactions.csv
│   └── social.json
├── output/                     # Pipeline outputs (git-ignored)
│   ├── entities.parquet
│   ├── events.parquet
│   ├── entity_graph.pkl
│   ├── risk_scores.parquet
│   └── ground_truth.json
└── tests/
    ├── test_ingest.py
    └── test_entity_resolution.py
```

---

## Reference Architecture

This prototype is inspired by the design patterns documented in:

- **Maltego** — link analysis + OSINT aggregation
- **Cognyte Nexyte** — telecom intelligence fusion
- **GraphAware Hume** — graph-native investigation platform
- **Quantexa** — entity resolution + network risk scoring

The four analytical primitives (temporal sequences, relationship graphs, geospatial traces, identity fingerprints) and the recommended stack (Python, NetworkX, scikit-learn, Streamlit) follow the survey's recommendations for academic/hackathon prototypes.

---

*FusionWatch MVP — built for demonstration purposes. Synthetic data only.*
