"""Experimental Jev-Omni multimodal decision adapter for llama.cpp.

The server must run the matching GGUF with --embedding --pooling none. This
adapter applies Jev-Omni's trained FP32 decision head to the unnormalized
last-token hidden state. Media input uses a matching Gemma 4 Unified mmproj.
"""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import Request, urlopen

import numpy as np


def post_json(url: str, payload: dict) -> object:
    request = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        return json.load(response)


def _video_frames(path: Path, count: int = 16) -> list[bytes]:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Could not count video frames: {path}")
    wanted = sorted({int(round((total - 1) * (k + .5) / count)) for k in range(count)})
    frames = []
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        buffer = BytesIO()
        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(buffer, format="PNG")
        frames.append(buffer.getvalue())
    cap.release()
    if not frames:
        raise ValueError(f"Could not decode video: {path}")
    return frames


def _audio_wav(path: Path) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-t", "30", "-ac", "1", "-ar", "16000", str(output)],
            check=True,
        )
        return output.read_bytes()


def decide(
    server: str, head_path: Path, state: str, question: str, options: list[str],
    image: Path | None = None, audio: Path | None = None, video: Path | None = None,
) -> dict:
    if not 2 <= len(options) <= 256:
        raise ValueError("Jev-Omni requires 2–256 options")
    if sum(value is not None for value in (image, audio, video)) > 1:
        raise ValueError("Provide at most one image, audio, or video file")
    choices = "\n".join(f"{i + 1}. {value}" for i, value in enumerate(options))
    text = (
        f"{state}\n\n---\n\nQUESTION: {question}\n\nOPTIONS:\n{choices}\n\n"
        f"Reply with only the number of the correct option (1-{len(options)}).\n"
        "Output a single number and nothing else."
    )
    # Match the pinned upstream Transformers chat template for one user turn.
    # llama.cpp's default /apply-template injects a system turn and omits the
    # final thought-channel prefix. The server inserts BOS automatically, so
    # sending a literal <bos> here would silently create two BOS tokens.
    if image is None and audio is None and video is None:
        prompt = f"<|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        response = post_json(server + "/embeddings", {"input": prompt})
    else:
        props = json.loads(urlopen(server + "/props", timeout=30).read())
        needed = "audio" if audio is not None else "video" if video is not None else "vision"
        if not props.get("modalities", {}).get(needed):
            raise ValueError(f"Server does not support {needed}")
        marker = props["media_marker"]
        media_bytes = [image.read_bytes()] if image is not None else [_audio_wav(audio)] if audio is not None else _video_frames(video)
        prompt = f"<|turn>user\n{marker * len(media_bytes)}{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
        media = [base64.b64encode(value).decode("ascii") for value in media_bytes]
        response = post_json(server + "/embedding", {
            "content": {"prompt_string": prompt, "multimodal_data": media},
            "embd_normalize": -1,
        })
    hidden = np.asarray(response[0]["embedding"][-1], dtype=np.float32)
    with np.load(head_path) as head:
        if hidden.shape != (3840,):
            raise ValueError(f"Expected 3840 hidden dimensions; got {hidden.shape}")
        z = head["linear.weight"][: len(options)] @ ((hidden - head["mu"][0]) / head["sd"][0])
        z += head["linear.bias"][: len(options)]
    z -= z.max()
    probabilities = np.exp(z)
    probabilities /= probabilities.sum()
    best = int(probabilities.argmax())
    return {
        "prediction": options[best],
        "prediction_index": best,
        "confidence": float(probabilities[best]),
        "probabilities": dict(zip(options, map(float, probabilities))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8080")
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    print(json.dumps(decide(
        args.server, args.head, args.state, args.question, args.options,
        image=args.image, audio=args.audio, video=args.video,
    ), indent=2))


if __name__ == "__main__":
    main()
