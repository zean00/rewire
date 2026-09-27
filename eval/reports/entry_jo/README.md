---
base_model: akhilaaa3/Jev-Omni
base_model_relation: quantized
license: apache-2.0
language:
  - en
  - fa
tags:
  - gguf
  - quantized
  - multimodal
  - decision-classification
model_name: Jev-Omni Q4_K_M GGUF
---

# Jev-Omni Q4_K_M GGUF

## Memory use

The quantized backbone is **6.87 GiB on disk** (7,381,382,848 bytes). The multimodal projector adds 116.38 MiB, and the required FP32 decision head adds 3.78 MiB.

| Stallion setup | Measured memory |
| --- | ---: |
| GPU, text-only model loaded | **8,980 MiB VRAM (8.77 GiB)** |
| GPU, model and multimodal projector loaded | **9,148 MiB VRAM (8.93 GiB)** |
| GPU, after an image decision | **9,256 MiB VRAM (9.04 GiB)** |
| CPU, after a text decision | **12.29 GiB process PSS** |
| CPU, after a 16-frame video decision | **13.49 GiB process PSS** |

GPU figures are `nvidia-smi` readings for the llama.cpp process on an RTX 5080 Laptop GPU with full offload and 2,048 context. CPU text used 1,024 context; CPU video used 8,192 context. These are observed readings, not maximums across all inputs and settings. Allow additional memory for the OS and other programs.

This is a [Q4_K_M quantization of `akhilaaa3/Jev-Omni`](https://huggingface.co/akhilaaa3/Jev-Omni), source revision `c050d51354147985d13286cf4acf90f562f2c631`. Its Gemma 4 12B backbone was converted with [llama.cpp](https://github.com/ggml-org/llama.cpp) revision `6e60f35608ec6918b44a9839c0c433687165f086` and quantized with `llama-quantize ... Q4_K_M 8`, without an importance matrix. The quantizer reported **4.95 bits per weight overall**; Q4_K_M is a mixed-precision format, not exactly four bits for every tensor. The multimodal projector and decision head retain their original precision.

**All three files and `jev_omni_gguf_decide.py` are needed for multimodal decisions.** For text-only decisions, the projector is optional. This is a classifier: it returns probabilities over supplied options, not generated answers. Plain `llama-cli` text generation does not apply the trained decision head.

## Measured speed

One benchmark request contained six sequential questions. These are end-to-end request timings, including six local adapter calls to the llama.cpp server, not token-generation rates or a matched-hardware comparison with Bev.

| Stallion configuration | Requests measured | Median per six-question request | Approx. time per decision |
| --- | ---: | ---: | ---: |
| AMD Ryzen AI 9 HX 370 CPU-only, 8 llama.cpp threads, 2,048 context | 23 | 11.30 s | 1.88 s |
| RTX 5080 Laptop GPU, full CUDA offload, 2,048 context | 106 | 1.18 s | 0.20 s |

The CPU subset came from an exploratory mixed-device run. The GPU timings came from a clean full GPU run. They are warm, local measurements on one machine, not guaranteed speed on other CPUs, GPUs, context sizes, or media types. Audio/video preprocessing and larger context can add substantial time.

## Quality checks

On the [Jev Persian Benchmark](https://github.com/ArmanJR/Jev-Persian-Benchmark) at revision `ac218d96630da9d9cc08fd897868c4d3c7048b0d`, the complete run returned **624/624 valid answers**: 480 main Persian questions, 48 English comparison answers, and 96 repeatability answers. The 106 figure in the runner is **batched requests**, not questions.

| Main Persian primitive | This Q4 | Bev baseline |
| --- | ---: | ---: |
| Choice | 237/240 (98.75%) | 229/240 (95.42%) |
| Noul | 150/160 (93.75%) | 152/160 (95.00%) |
| Score, within ±0.5 rubric level | 76/80 (95.00%) | 70/80 (87.50%) |

Jev-Omni Q4 leads on Choice and Score; Bev leads slightly on Noul. Both missed six of the same main questions. See the [paired report](persian-report.md) and [aggregate results](comparison.json). These models use different native prompts and output mechanisms, so these are end-to-end system results rather than an isolated quantization-loss estimate. Neither model changed a decision across the benchmark's 48 repeat groups. For the original Jev-Omni FP32 source, this Q4 matched the winning option on **3/4** published text verification examples; the maximum absolute probability difference was **0.210**. **Do not assume source-equivalent probabilities or calibration.** The [verification vectors](verify-q4-unified.json) document the discrepancy. Image, audio, and video routes passed small smoke checks, not accuracy benchmarks.

## Download and run

Install a [current llama.cpp build](https://github.com/ggml-org/llama.cpp) with `llama-server` and Python 3 with NumPy (`python -m pip install numpy`). The pinned revision above is the tested version; newer builds may need their embedding and multimodal behavior checked. FFmpeg is needed for audio input; OpenCV and Pillow are needed for video input. These commands are written on single lines so they work in a shell or can be adapted to Windows PowerShell.

```bash
hf download Reza2kn/Jev-Omni-Q4_K_M-GGUF --local-dir jev-omni-q4
cd jev-omni-q4
llama-server -m Jev-Omni-Unified-Q4_K_M.gguf --mmproj mmproj-jev-omni.gguf --embedding --pooling none --host 127.0.0.1 --port 8080 --ctx-size 8192 --parallel 1 --batch-size 2048 --ubatch-size 2048 --threads 8 --no-warmup -ngl 0
```

In another terminal, from `jev-omni-q4`:

```bash
python jev_omni_gguf_decide.py --server http://127.0.0.1:8080 --head decision-head-f32.npz --state "The meeting starts at 10 AM. It is now 9 AM." --question "Has the meeting started?" --options Yes No
```

For supported GPU offload, replace `-ngl 0` with `-ngl 99` or use a smaller layer count if the GPU cannot hold the model. For text-only use, `--mmproj` can be omitted. The adapter also accepts `--image FILE`, `--audio FILE`, or `--video FILE` with a loaded projector. Linux CPU and Stallion CUDA were tested. The GGUF/llama.cpp approach is intended to be portable, but this specific adapter has **not** been validated on macOS, Windows, Metal, or Vulkan. Keep the server bound to loopback: the embedding endpoint exposes internal model states.

## Files and provenance

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `Jev-Omni-Unified-Q4_K_M.gguf` | quantized backbone | `35bf51cb0dee2504e9f16093278ca853da2888251577236371630124605c40cd` |
| `mmproj-jev-omni.gguf` | original-precision multimodal projector | `44e54654d861a2f1dc90a8a787f6a8ea6f96b886e79cd6866b167a7fb823c668` |
| `decision-head-f32.npz` | original-precision decision head | `47b346e120dc9110ef7c0b610cf5d1f459ddcdb05ffd3cd126b6938d88a8bb77` |

The four exported head tensors were checked bit-for-bit against the source `head.pt`. The source model is Apache-2.0 licensed; benchmark data rights remain with their respective authors. This is an independent quantization of Jev-Omni, not a copy of or affiliation with the API-only Jev service.
