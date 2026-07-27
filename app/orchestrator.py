from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent


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

    def analyze(self, transcript: str) -> dict[str, Any]:
        if not transcript.strip():
            raise ValueError("Transcript must not be empty.")

        with ThreadPoolExecutor(max_workers=4) as executor:
            classifier_future = executor.submit(
                self._classifier.analyze,
                transcript,
            )
            quality_future = executor.submit(
                self._quality.analyze,
                transcript,
            )
            compliance_future = executor.submit(
                self._compliance.analyze,
                transcript,
            )
            summarizer_future = executor.submit(
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