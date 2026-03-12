# BlackPanther V2

**Autonomous AI Penetration Testing Agent** — A mathematical-proof-driven ethical hacking framework that couples differential-equation models with HJB optimal control to balance attack intensity against detection risk in real time.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

BlackPanther V2 is an autonomous penetration testing system that:

- **Reconnaissance** — Discovers hosts and services (nmap) or crawls web applications
- **Vulnerability Scanning** — Identifies CVEs and OWASP Top 10 web vulnerabilities
- **Exploitation** — Generates exploit scripts via LLM (DeepSeek/Mistral)
- **Optimal Control** — Uses Hamilton-Jacobi-Bellman (HJB) to balance attack vs. stealth

The system is driven by four coupled mathematical models:

| Model | Type | Role |
|-------|------|------|
| **Knowledge Evolution** | ODE | Attacker learning rate grows with activity |
| **Suspicion Field** | PDE (reaction-diffusion) | Defender awareness spreads through the network |
| **Access Propagation** | Network ODE | Access spreads across hosts like an epidemic |
| **HJB Controller** | Optimal control | Chooses attack intensity and stealth dynamically |

---

## Quick Start

### Prerequisites

- Python 3.9+
- [nmap](https://nmap.org/) (for network scanning)
- Redis (optional, for Celery/WebSocket in API mode)

### Installation

```bash
# Clone and enter project
cd blackpanther-v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install package
pip install -e .
