from prometheus_client import Counter, Gauge, Histogram

CALLS_TOTAL = Counter(
    "mtbank_calls_total",
    "Number of completed call analyses.",
    ["topic", "priority", "compliance_passed"],
)

CALL_ERRORS_TOTAL = Counter(
    "mtbank_call_errors_total",
    "Number of failed call analyses.",
    ["error_type"],
)

CALL_QUALITY_SCORE = Histogram(
    "mtbank_call_quality_score",
    "Quality score returned by the quality agent.",
    buckets=(0, 25, 50, 75, 100),
)

CALL_QUALITY_LATEST = Gauge(
    "mtbank_call_quality_latest",
    "Quality score of the most recently completed call.",
)

CALL_ANALYSIS_DURATION_SECONDS = Histogram(
    "mtbank_call_analysis_duration_seconds",
    "End-to-end call analysis duration.",
    buckets=(1, 3, 10, 30, 60, 120, 300),
)

REALTIME_CHUNKS_TOTAL = Counter(
    "mtbank_realtime_chunks_total",
    "Number of real-time audio chunks processed.",
    ["status"],
)

REALTIME_CHUNK_PROCESSING_SECONDS = Histogram(
    "mtbank_realtime_chunk_processing_seconds",
    "Processing latency for a real-time audio chunk.",
    buckets=(0.1, 0.25, 0.5, 1, 2, 3, 5, 10),
)

TREND_BATCHES_TOTAL = Counter(
    "mtbank_trend_batches_total",
    "Number of completed multi-call trend analyses.",
)


def record_call_analysis(
    analysis: dict[str, object],
    duration_seconds: float,
) -> None:
    classification = analysis.get("classification", {})
    quality_score = analysis.get("quality_score", {})
    compliance = analysis.get("compliance", {})

    classification = classification if isinstance(classification, dict) else {}
    quality_score = quality_score if isinstance(quality_score, dict) else {}
    compliance = compliance if isinstance(compliance, dict) else {}

    topic = str(classification.get("topic", "не определено"))
    priority = str(classification.get("priority", "unknown"))
    compliance_passed = str(bool(compliance.get("passed", False))).lower()

    try:
        score = float(quality_score.get("total", 0))
    except (TypeError, ValueError):
        score = 0.0

    CALLS_TOTAL.labels(
        topic=topic,
        priority=priority,
        compliance_passed=compliance_passed,
    ).inc()
    CALL_QUALITY_SCORE.observe(score)
    CALL_QUALITY_LATEST.set(score)
    CALL_ANALYSIS_DURATION_SECONDS.observe(duration_seconds)
