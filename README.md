# BlackPanther V2

Autonomous AI Penetration Testing Agent — mathematical models for ethical hacking with HJB optimal control.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is BlackPanther?

BlackPanther is an autonomous penetration testing system that:

- **Reconnoitres** targets (network via nmap, web apps via crawling)
- **Scans** for vulnerabilities (CVEs, OWASP Top 10)
- **Exploits** findings by generating Python exploit scripts via LLM (DeepSeek/Mistral)
- **Adapts** using four coupled differential-equation models and HJB optimal control

---

## Mathematical Models

**1. Knowledge Evolution (ODE)**  
`dK/dt = αK(1-K/K_max) - βK + γS + σξ`  
Learning grows with activity, decays over time, and benefits from defender reactions.

**2. Suspicion Field (PDE)**  
`∂S/∂t = D∇²S + rS(1-S) - δKA + σξ`  
Defender awareness spreads across the network; attacks can suppress it.

**3. Access Propagation (ODE)**  
`dA/dt = ηKA(1-A) - μA + lateral spread`  
Access spreads across hosts like an epidemic.

**4. HJB Controller**  
Chooses attack intensity and stealth to maximize gain while limiting detection risk.

---

## Quick Start

**Prerequisites:** Python 3.9+, [nmap](https://nmap.org/)

```bash
git clone <repo-url>
cd blackpanther-v2
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

**Configure** — create `.env` in project root:

```env
DEEPSEEK_API_KEY=sk-your-key
# or: LLM_PROVIDER=mistral + MISTRAL_API_KEY=...
```

**Run a scan:**

```bash
python -m blackpanther.agents.coordinator https://juice-shop.herokuapp.com
python -m blackpanther.agents.coordinator 192.168.1.1
```

**Mathematical demo (no network):**

```bash
python examples/mathematical_demo.py
```

---

## Output

| Path | Contents |
|------|----------|
| `output/proofs/` | Plots (knowledge, suspicion, access, HJB policy) |
| `output/exploits/` | Generated exploit scripts (manual review required) |
| `output/web_reports/` | JSON vulnerability reports |

---

## API Mode

```bash
redis-server
celery -A blackpanther.api.django_settings worker -l info
blackpanther-api runserver 0.0.0.0:8000
```

POST to `/api/v1/scan/start` with `{"target": "https://example.com"}`.

---
## WARNING!!!
- Don't test it without any legal written permissions!
- LLMs inside this are fine-tuned specially for generating real exploits! So proceed with legal permissions!
## License

MIT © BlackPanther Research Team
