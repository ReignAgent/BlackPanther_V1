"""GPT-3.5-Turbo Interactive Report Generator.

Generates comprehensive penetration testing reports with:
  - Executive summary for stakeholders
  - Technical details for security teams
  - Risk assessment with CVSS scoring
  - Remediation recommendations
  - Interactive Q&A for follow-up questions

Uses OpenAI's GPT-3.5-turbo (free tier compatible).
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from blackpanther.settings import Settings, get_settings

from .interfaces import LLMProvider
from .resilience import async_retry


EXECUTIVE_SUMMARY_PROMPT = textwrap.dedent("""\
    You are a senior cybersecurity consultant writing an executive summary for a penetration test report.
    
    Write a clear, concise executive summary (3-4 paragraphs) that:
    1. Summarizes the scope and objectives of the assessment
    2. Highlights critical findings and their business impact
    3. Provides overall risk rating (Critical/High/Medium/Low)
    4. Recommends immediate actions for leadership
    
    Use professional language suitable for C-level executives.
    Do not include technical jargon or CVE details.
""")

TECHNICAL_DETAILS_PROMPT = textwrap.dedent("""\
    You are a penetration testing expert writing the technical findings section of a security report.
    
    For each vulnerability, provide:
    1. Vulnerability title and CVE ID
    2. Affected systems/services
    3. Technical description of the issue
    4. Proof of concept or exploitation steps (sanitized)
    5. CVSS score interpretation
    
    Use clear technical language suitable for IT security teams.
    Format using markdown headers and bullet points.
""")

RISK_ASSESSMENT_PROMPT = textwrap.dedent("""\
    You are a risk analyst creating a risk assessment based on penetration test findings.
    
    Provide:
    1. Overall risk rating with justification
    2. Risk matrix showing likelihood vs impact
    3. Business impact analysis for each critical finding
    4. Compliance implications (GDPR, PCI-DSS, HIPAA if applicable)
    5. Prioritized risk remediation timeline
    
    Use a structured format with clear risk ratings.
""")

RECOMMENDATIONS_PROMPT = textwrap.dedent("""\
    You are a security architect providing remediation recommendations.
    
    For each vulnerability or finding, provide:
    1. Short-term fixes (immediate actions)
    2. Long-term solutions (architectural changes)
    3. Preventive measures to avoid similar issues
    4. Relevant security controls and frameworks
    5. Estimated effort/complexity for remediation
    
    Prioritize recommendations by risk level and ease of implementation.
""")

INTERACTIVE_QA_PROMPT = textwrap.dedent("""\
    You are an AI security assistant helping to answer questions about a penetration test report.
    
    You have access to:
    - The scan results including hosts, vulnerabilities, and exploits
    - The generated report sections
    
    Answer the user's question clearly and concisely.
    If the question cannot be answered from the available data, say so.
    Provide specific details and recommendations when relevant.
""")


class ReportGenerator:
    """Generate interactive penetration test reports using GPT-3.5-turbo.
    
    Usage:
        generator = ReportGenerator(settings)
        summary = await generator.generate_executive_summary(results)
        details = await generator.generate_technical_details(vulns)
        
        # Interactive Q&A
        answer = await generator.interactive_query("What's the most critical finding?", context)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        self._api_key = cfg.openai_api_key
        self._base_url = cfg.openai_base_url
        self._model = cfg.report_model
        self._max_tokens = cfg.report_max_tokens
        self._temperature = cfg.report_temperature
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise RuntimeError("openai package required: pip install openai") from exc
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    @async_retry(max_attempts=3, backoff=2.0)
    async def _generate(self, system_prompt: str, user_content: str) -> str:
        """Generate text using GPT-3.5-turbo."""
        if not self._api_key:
            logger.warning("[report] No OpenAI API key — returning placeholder")
            return self._placeholder_response(system_prompt)
        
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""

    def _placeholder_response(self, prompt_type: str) -> str:
        """Return placeholder when API is unavailable."""
        return textwrap.dedent(f"""\
            ## Report Section (API Unavailable)
            
            This section could not be generated because the OpenAI API key is not configured.
            
            To enable report generation:
            1. Set the `OPENAI_API_KEY` environment variable
            2. Or add it to your `.env` file
            
            The GPT-3.5-turbo model is used for report generation and is available on OpenAI's free tier.
        """)

    async def generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary for stakeholders."""
        context = self._build_summary_context(results)
        return await self._generate(EXECUTIVE_SUMMARY_PROMPT, context)

    async def generate_technical_details(self, vulns: List[Dict[str, Any]]) -> str:
        """Generate technical findings section."""
        context = self._build_vulns_context(vulns)
        return await self._generate(TECHNICAL_DETAILS_PROMPT, context)

    async def generate_risk_assessment(self, results: Dict[str, Any]) -> str:
        """Generate risk assessment section."""
        context = self._build_risk_context(results)
        return await self._generate(RISK_ASSESSMENT_PROMPT, context)

    async def generate_recommendations(self, exploits: List[Dict[str, Any]]) -> str:
        """Generate remediation recommendations."""
        context = self._build_recommendations_context(exploits)
        return await self._generate(RECOMMENDATIONS_PROMPT, context)

    async def interactive_query(self, question: str, context: Dict[str, Any]) -> str:
        """Answer interactive questions about the report."""
        full_context = self._build_qa_context(question, context)
        return await self._generate(INTERACTIVE_QA_PROMPT, full_context)

    async def generate_full_report(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Generate all report sections."""
        vulns = results.get("vulns", [])
        exploits = results.get("exploits", [])

        logger.info("[report] Generating executive summary...")
        executive_summary = await self.generate_executive_summary(results)

        logger.info("[report] Generating technical details...")
        technical_details = await self.generate_technical_details(vulns)

        logger.info("[report] Generating risk assessment...")
        risk_assessment = await self.generate_risk_assessment(results)

        logger.info("[report] Generating recommendations...")
        recommendations = await self.generate_recommendations(exploits)

        return {
            "executive_summary": executive_summary,
            "technical_details": technical_details,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat(),
            "model_used": self._model,
        }

    def _build_summary_context(self, results: Dict[str, Any]) -> str:
        """Build context for executive summary generation."""
        hosts = results.get("hosts", [])
        vulns = results.get("vulns", [])
        exploits = results.get("exploits", [])
        duration = results.get("duration", 0)

        critical_count = sum(1 for v in vulns if isinstance(v, dict) and v.get("severity", 0) >= 9.0)
        high_count = sum(1 for v in vulns if isinstance(v, dict) and 7.0 <= v.get("severity", 0) < 9.0)

        return textwrap.dedent(f"""\
            Penetration Test Results Summary:
            
            - Target Systems: {len(hosts)} hosts scanned
            - Hosts: {', '.join(hosts[:5])}{'...' if len(hosts) > 5 else ''}
            - Total Vulnerabilities: {len(vulns)}
              - Critical (CVSS >= 9.0): {critical_count}
              - High (CVSS 7.0-8.9): {high_count}
            - Exploits Successfully Generated: {len(exploits)}
            - Assessment Duration: {duration:.1f} seconds
            
            Top Vulnerabilities by Severity:
            {self._format_top_vulns(vulns[:5])}
            
            Please generate an executive summary based on these findings.
        """)

    def _build_vulns_context(self, vulns: List[Dict[str, Any]]) -> str:
        """Build context for technical details generation."""
        if not vulns:
            return "No vulnerabilities were discovered during this assessment."

        vuln_details = []
        for v in vulns[:20]:
            if isinstance(v, dict):
                vuln_details.append(
                    f"- {v.get('cve', 'Unknown')} (CVSS: {v.get('severity', 'N/A')})\n"
                    f"  Service: {v.get('service', 'Unknown')} on port {v.get('port', 'N/A')}\n"
                    f"  Title: {v.get('title', 'No title')}"
                )
            else:
                vuln_details.append(f"- {v}")

        return textwrap.dedent(f"""\
            Discovered Vulnerabilities ({len(vulns)} total):
            
            {chr(10).join(vuln_details)}
            
            Please provide detailed technical analysis for each vulnerability.
        """)

    def _build_risk_context(self, results: Dict[str, Any]) -> str:
        """Build context for risk assessment generation."""
        vulns = results.get("vulns", [])
        exploits = results.get("exploits", [])
        
        knowledge = results.get("knowledge_final", 0)
        suspicion = results.get("suspicion_mean", 0)
        access = results.get("access_global", 0)

        return textwrap.dedent(f"""\
            Risk Assessment Data:
            
            Vulnerability Statistics:
            - Total vulnerabilities: {len(vulns)}
            - Exploitable vulnerabilities: {len(exploits)}
            - Exploitation success rate: {len(exploits) / max(len(vulns), 1) * 100:.1f}%
            
            System State Metrics (from mathematical model):
            - Knowledge accumulated: {knowledge:.2f}
            - Mean suspicion level: {suspicion:.4f}
            - Global access level: {access:.3f}
            
            Vulnerability Severity Distribution:
            {self._severity_distribution(vulns)}
            
            Please provide a comprehensive risk assessment.
        """)

    def _build_recommendations_context(self, exploits: List[Dict[str, Any]]) -> str:
        """Build context for recommendations generation."""
        if not exploits:
            return textwrap.dedent("""\
                No exploits were successfully generated during this assessment.
                
                This could indicate:
                1. Strong security posture
                2. Limited scope of the assessment
                3. Need for manual testing
                
                Please provide general security recommendations.
            """)

        exploit_details = []
        for e in exploits[:10]:
            if isinstance(e, dict):
                exploit_details.append(
                    f"- {e.get('cve', 'Unknown')}\n"
                    f"  Aggressiveness: {e.get('aggressiveness', 'N/A')}\n"
                    f"  Code lines: {e.get('code_lines', 'N/A')}"
                )
            else:
                exploit_details.append(f"- {e}")

        return textwrap.dedent(f"""\
            Successfully Generated Exploits ({len(exploits)} total):
            
            {chr(10).join(exploit_details)}
            
            Please provide prioritized remediation recommendations for these findings.
        """)

    def _build_qa_context(self, question: str, context: Dict[str, Any]) -> str:
        """Build context for interactive Q&A."""
        results = context.get("results", {})
        report = context.get("report", {})

        return textwrap.dedent(f"""\
            Context Information:
            
            Scan Results:
            - Hosts: {len(results.get('hosts', []))}
            - Vulnerabilities: {len(results.get('vulns', []))}
            - Exploits: {len(results.get('exploits', []))}
            
            Report Sections Available:
            - Executive Summary: {'Yes' if report.get('executive_summary') else 'No'}
            - Technical Details: {'Yes' if report.get('technical_details') else 'No'}
            - Risk Assessment: {'Yes' if report.get('risk_assessment') else 'No'}
            - Recommendations: {'Yes' if report.get('recommendations') else 'No'}
            
            User Question: {question}
            
            Please answer the question based on the available information.
        """)

    def _format_top_vulns(self, vulns: List[Dict[str, Any]]) -> str:
        """Format top vulnerabilities for context."""
        if not vulns:
            return "No vulnerabilities found."
        
        lines = []
        for v in vulns:
            if isinstance(v, dict):
                lines.append(f"- {v.get('cve', 'Unknown')}: {v.get('title', 'No title')} (CVSS: {v.get('severity', 'N/A')})")
            else:
                lines.append(f"- {v}")
        return "\n".join(lines)

    def _severity_distribution(self, vulns: List[Dict[str, Any]]) -> str:
        """Calculate severity distribution."""
        critical = high = medium = low = info = 0
        
        for v in vulns:
            if isinstance(v, dict):
                sev = v.get("severity", 0)
                if sev >= 9.0:
                    critical += 1
                elif sev >= 7.0:
                    high += 1
                elif sev >= 4.0:
                    medium += 1
                elif sev >= 0.1:
                    low += 1
                else:
                    info += 1

        return textwrap.dedent(f"""\
            - Critical (9.0-10.0): {critical}
            - High (7.0-8.9): {high}
            - Medium (4.0-6.9): {medium}
            - Low (0.1-3.9): {low}
            - Informational: {info}
        """)


class StubReportGenerator(ReportGenerator):
    """Stub report generator for testing without API access."""

    async def _generate(self, system_prompt: str, user_content: str) -> str:
        return textwrap.dedent("""\
            ## Stub Report Section
            
            This is a placeholder report generated without API access.
            
            In a production environment with configured API keys, this section would contain:
            - AI-generated analysis of the penetration test results
            - Detailed findings and recommendations
            - Risk assessment and remediation guidance
            
            Configure OPENAI_API_KEY to enable full report generation.
        """)
