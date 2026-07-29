"""Send a PCM WAV to the real-time WebSocket endpoint.

Usage:
    python scripts/realtime_smoke.py test_data/asr_smoke.wav
"""

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path

import websockets


async def run(audio_path: Path, url: str) -> None:
    with wave.open(str(audio_path), "rb") as audio_file:
        if audio_file.getnchannels() != 1:
            raise ValueError("The smoke file must be mono.")
        if audio_file.getsampwidth() != 2:
            raise ValueError("The smoke file must be PCM16.")

        sample_rate = audio_file.getframerate()
        frames_per_chunk = sample_rate
        chunks: list[bytes] = []

        while True:
            frames = audio_file.readframes(frames_per_chunk)
            if not frames:
                break
            chunks.append(frames)

    async with websockets.connect(url, max_size=10 * 1024 * 1024) as socket:
        ready = json.loads(await socket.recv())
        print(json.dumps(ready, ensure_ascii=False))
        await socket.send(
            json.dumps(
                {
                    "type": "start",
                    "sample_rate": sample_rate,
                    "language": "ru",
                }
            )
        )
        print(json.dumps(json.loads(await socket.recv()), ensure_ascii=False))

        latencies: list[float] = []

        for chunk in chunks:
            sent_at = time.perf_counter()
            await socket.send(chunk)

            if len(chunk) < frames_per_chunk * 2:
                await socket.send(
                    json.dumps({"type": "flush"})
                )

            response = json.loads(await socket.recv())
            round_trip_ms = (time.perf_counter() - sent_at) * 1000
            latencies.append(round_trip_ms)
            print(
                json.dumps(
                    {
                        "type": response.get("type"),
                        "processing_ms": response.get("processing_ms"),
                        "round_trip_ms": round(round_trip_ms, 2),
                        "latency_target_met": response.get(
                            "latency_target_met"
                        ),
                        "text": response.get("text", ""),
                    },
                    ensure_ascii=False,
                )
            )

        await socket.send(json.dumps({"type": "stop"}))
        completed = json.loads(await socket.recv())
        print(json.dumps(completed, ensure_ascii=False))

        if latencies:
            print(
                json.dumps(
                    {
                        "max_round_trip_ms": round(max(latencies), 2),
                        "all_under_3000ms": max(latencies) < 3000,
                    },
                    ensure_ascii=False,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/transcribe",
    )
    args = parser.parse_args()
    asyncio.run(run(args.audio, args.url))


if __name__ == "__main__":
    main()
