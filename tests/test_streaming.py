from asr.streaming import StreamingTranscriber


def test_streaming_detects_silence() -> None:
    assert StreamingTranscriber._is_silence(
        b"\x00\x00" * 16000
    )


def test_streaming_does_not_drop_audible_pcm() -> None:
    loud_sample = (2000).to_bytes(
        2,
        byteorder="little",
        signed=True,
    )

    assert not StreamingTranscriber._is_silence(
        loud_sample * 16000
    )
