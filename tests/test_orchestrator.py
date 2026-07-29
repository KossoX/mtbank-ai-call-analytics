from typing import Any

import pytest

from agents.classifier import ClassifierAgent
from agents.compliance import ComplianceAgent
from agents.quality import QualityAgent
from agents.summarizer import SummarizerAgent
from app.orchestrator import AnalysisOrchestrator


class FakeClassifierAgent(ClassifierAgent):
    def __init__(self) -> None:
        pass

    def analyze(self, transcript: str) -> dict[str, str]:
        return {
            "topic": "кредиты",
            "priority": "medium",
        }


class FakeQualityAgent(QualityAgent):
    def __init__(self) -> None:
        pass

    def analyze(self, transcript: str) -> dict[str, Any]:
        return {
            "total": 75,
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": False,
            },
        }


class FakeComplianceAgent(ComplianceAgent):
    def __init__(self) -> None:
        pass

    def analyze(self, transcript: str) -> dict[str, Any]:
        return {
            "passed": True,
            "issues": [],
        }


class FakeSummarizerAgent(SummarizerAgent):
    def __init__(self) -> None:
        pass

    def analyze(self, transcript: str) -> dict[str, Any]:
        return {
            "summary": "Краткое резюме звонка.",
            "action_items": ["Отправить инструкцию клиенту."],
        }


def test_orchestrator_combines_agent_results() -> None:
    orchestrator = AnalysisOrchestrator(
        classifier=FakeClassifierAgent(),
        quality=FakeQualityAgent(),
        compliance=FakeComplianceAgent(),
        summarizer=FakeSummarizerAgent(),
    )

    result = orchestrator.analyze("Тестовый транскрипт.")

    assert result == {
        "classification": {
            "topic": "кредиты",
            "priority": "medium",
        },
        "quality_score": {
            "total": 75,
            "checklist": {
                "greeting": True,
                "need_detection": True,
                "solution_provided": True,
                "farewell": False,
            },
        },
        "compliance": {
            "passed": True,
            "issues": [],
        },
        "summary": "Краткое резюме звонка.",
        "action_items": [
            "Отправить инструкцию клиенту.",
        ],
    }


def test_orchestrator_rejects_empty_transcript() -> None:
    orchestrator = AnalysisOrchestrator(
        classifier=FakeClassifierAgent(),
        quality=FakeQualityAgent(),
        compliance=FakeComplianceAgent(),
        summarizer=FakeSummarizerAgent(),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        orchestrator.analyze("")
