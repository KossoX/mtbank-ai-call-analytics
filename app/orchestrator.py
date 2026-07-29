from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from app.json_logging import log_agent_event

AgentCallable = Callable[[str], Any]


class AnalysisOrchestrator:
    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        quality: QualityAgent | None = None,
        compliance: ComplianceAgent | None = None,
        summarizer: SummarizerAgent | None = None,
    ) -> None:
        self._classifier = classifier or ClassifierAgent()
        self._quality = quality or QualityAgent()
        self._compliance = compliance or ComplianceAgent()
        self._summarizer = summarizer or SummarizerAgent()

    @staticmethod
    def _run_agent(
        agent_name: str,
        agent_callable: AgentCallable,
        transcript: str,
    ) -> Any:
        log_agent_event(
            agent=agent_name,
            event="agent.input",
            payload={"transcript": transcript},
        )

        started_at = perf_counter()

        try:
            result = agent_callable(transcript)
        except Exception as error:
            duration_ms = (perf_counter() - started_at) * 1000

            log_agent_event(
                agent=agent_name,
                event="agent.error",
                payload={
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                duration_ms=duration_ms,
            )

            raise

        duration_ms = (perf_counter() - started_at) * 1000

        log_agent_event(
            agent=agent_name,
            event="agent.output",
            payload=result,
            duration_ms=duration_ms,
        )

        return result

    def analyze(self, transcript: str) -> dict[str, Any]:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty.")

        with ThreadPoolExecutor(max_workers=4) as executor:
            classifier_future = executor.submit(
                self._run_agent,
                "classifier",
                self._classifier.analyze,
                transcript,
            )
            quality_future = executor.submit(
                self._run_agent,
                "quality",
                self._quality.analyze,
                transcript,
            )
            compliance_future = executor.submit(
                self._run_agent,
                "compliance",
                self._compliance.analyze,
                transcript,
            )
            summarizer_future = executor.submit(
                self._run_agent,
                "summarizer",
                self._summarizer.analyze,
                transcript,
            )

            classification = classifier_future.result()
            quality_score = quality_future.result()
            compliance = compliance_future.result()
            summary_result = summarizer_future.result()

        return {
            "classification": classification,
            "quality_score": quality_score,
            "compliance": compliance,
            "summary": summary_result["summary"],
            "action_items": summary_result["action_items"],
        }
