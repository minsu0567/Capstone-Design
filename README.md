# Capstone-Design

Voice-controlled RC car — Korean **standard and dialect** speech recognition with a fine-tuned
Whisper, and Llama 3.2 command parsing.

Speech goes to Whisper, the transcript goes to Llama 3.2, and the model emits a JSON command
list that a Jetson Orin Nano executes on the vehicle.

```
"천천히 회전해주라이"  ->  {"commands": [{"command_id": "slow", "value": -1, "duration": -1}]}
```

## Models

| Stage | Hugging Face |
|---|---|
| Whisper base | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) |
| Whisper — dialect ASR | [minsu0567/Capstone-Design-Whisper-Dialect](https://huggingface.co/minsu0567/Capstone-Design-Whisper-Dialect) |
| Llama 3.2 base | [Bllossom/llama-3.2-Korean-Bllossom-3B](https://huggingface.co/Bllossom/llama-3.2-Korean-Bllossom-3B) |
| Llama 3.2 — command parsing | [minsu0567/Capstone-Design-Llama3.2-Command](https://huggingface.co/minsu0567/Capstone-Design-Llama3.2-Command) |

Whisper is fine-tuned with QLoRA (4bit NF4, r=16, `q/k/v/out_proj`) and published with the
adapter already merged, so `from_pretrained` is enough. Llama 3.2 is published merged as well.

## Commands

The LLM emits `command_id`, `value` and `duration` per command; `-1` means "unset".

| id | 동작 | id | 동작 |
|---|---|---|---|
| `forward` | 직진 | `up` | 속도 올려 |
| `backward` | 후진 | `down` | 속도 내려 |
| `left` | 좌회전 | `fast` | 빨리 회전 |
| `right` | 우회전 | `slow` | 천천히 회전 |
| `stop` | 멈춤 | `setting` | 속도 설정 |
| `backward left` / `backward right` | 후진 회전 | `quit` | 종료 |

## Layout

```
capstone_design/
├── notebooks/                          Colab drivers (see below)
├── whisper_src/
│   ├── whisper_sft.py                  QLoRA fine-tuning, dataset split, WER metric
│   └── whisper_eval.py                 base vs fine-tuned WER/CER comparison
├── pipeline_src/
│   └── voice_command_pipeline.py       transcribe -> command -> HTTP, Gradio interface
├── llm_src/
│   └── llama3_2_sft.py                 Alpaca-format SFT for command parsing
├── device_src/
│   ├── control.py                      Jetson: ODrive motors, PCA9685 servos, video stream
│   └── local_edge_communication.py     Flask relay between Colab and the Jetson socket
└── data/
    ├── total_whisper.json              1796 ASR samples (standard 687 + dialect 1109)
    ├── total_commands.jsonl            command-parsing training set
    ├── test.jsonl                      held-out command set (100, no overlap with training)
    ├── command_lexicon.py              command vocabulary
    ├── generate_*_commands.py          synthetic command generators
    └── generate_test_set.py            builds test.jsonl and strips it from training
```

## Notebooks

| # | Notebook | Purpose |
|---|---|---|
| 1 | `whisper_finetuning.ipynb` | Whisper QLoRA fine-tuning on standard + dialect speech |
| 2 | `whisper_evaluation.ipynb` | base vs fine-tuned WER/CER on held-out dialect audio |
| 3 | `llama3_2_fine_tuning.ipynb` | Llama 3.2 SFT for speech-to-command parsing |
| 4 | `llama3_2_evaluation.ipynb` | exact-match accuracy on the held-out command set |
| 5 | `colab_local_communication.ipynb` | load both models and serve Gradio to the car |

## Google Drive

The notebooks run on Colab and read everything from Drive. Copy this repository to
`MyDrive/capstone_design/`; these must exist alongside it:

```
MyDrive/capstone_design/
├── data/audio_files/                            1796 mp3 files (not in this repository)
├── whisper-merged2/                             Whisper base for inference
├── whisper-dialect-turbo-qlora/checkpoint-1880/ QLoRA adapter
└── llama3.2-merged/                             merged Llama 3.2
```

`whisper_evaluation.ipynb` additionally reads `MyDrive/whisper_test/` (audio plus
`updated_test_data.json`), which sits next to `capstone_design/`, not inside it.

`data/audio_files/` is 256MB and stays out of git — `data/total_whisper.json` references the
files by name, so drop them into that one flat directory.

## Hardware

Jetson Orin Nano with ODrive brushless motors and a PCA9685 servo driver. `control.py` runs on
the car and listens on port 8001 for commands and 8000 for the video stream;
`local_edge_communication.py` runs on a machine that can reach both Colab and the car, and
forwards the JSON the Gradio app posts.
