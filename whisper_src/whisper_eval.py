import gc
import json
import os
import re
import unicodedata

import evaluate
import torch
from datasets import Audio, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

JSON_ENCODINGS = ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr')

_PUNCT = re.compile(r'[^\w\s]|_', flags=re.UNICODE)
_SPACE = re.compile(r'\s+')


def normalize_text(text):
    text = unicodedata.normalize('NFC', text)
    text = _PUNCT.sub(' ', text)
    return _SPACE.sub(' ', text).strip()


def load_test_dataset(json_path, audio_dir=None, sampling_rate=16000):
    data = None
    for encoding in JSON_ENCODINGS:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    if data is None:
        raise ValueError(f'JSON 파일을 읽을 수 없습니다: {json_path}')

    if audio_dir is not None:
        for item in data:
            item['audio'] = {'path': os.path.join(
                audio_dir, os.path.basename(item['audio']['path']))}

    missing = [item['audio']['path'] for item in data
               if not os.path.isfile(item['audio']['path'])]
    if missing:
        raise FileNotFoundError(
            f'오디오 파일 {len(missing)}개를 찾을 수 없습니다. 예: {missing[0]}')

    dataset = Dataset.from_list(data)
    return dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))


def load_whisper(model_id, device=None, torch_dtype=None,
                 language='korean', task='transcribe'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype)
    model.generation_config.language = language
    model.generation_config.task = task
    model.to(device)
    model.eval()

    processor = WhisperProcessor.from_pretrained(model_id, language=language, task=task)
    print(f'Loaded {model_id} on {device} ({torch_dtype})')
    return model, processor


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _feature_dtype(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            return module.weight.dtype
    return torch.float32


@torch.no_grad()
def transcribe(model, processor, dataset, batch_size=8, max_new_tokens=256,
               language='korean', task='transcribe', log_every=2):
    model.eval()
    device = next(model.parameters()).device
    dtype = _feature_dtype(model)

    predictions = []
    total = (len(dataset) + batch_size - 1) // batch_size
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        features = processor.feature_extractor(
            [audio['array'] for audio in batch['audio']],
            sampling_rate=batch['audio'][0]['sampling_rate'],
            return_tensors='pt',
        ).input_features.to(device=device, dtype=dtype)

        generated = model.generate(
            input_features=features,
            max_new_tokens=max_new_tokens,
            language=language,
            task=task,
            num_beams=1,
            do_sample=False,
        )
        predictions.extend(processor.batch_decode(generated, skip_special_tokens=True))

        step = start // batch_size + 1
        if log_every and step % log_every == 0:
            print(f'  추론 진행 중: {step}/{total} 배치')

    return predictions


def compute_scores(predictions, references):
    wer_metric = evaluate.load('wer')
    cer_metric = evaluate.load('cer')

    norm_pred = [normalize_text(p) for p in predictions]
    norm_ref = [normalize_text(r) for r in references]

    keep = [i for i, ref in enumerate(norm_ref) if ref]
    norm_pred = [norm_pred[i] for i in keep]
    norm_ref = [norm_ref[i] for i in keep]

    return {
        'wer': 100 * wer_metric.compute(predictions=norm_pred, references=norm_ref),
        'cer': 100 * cer_metric.compute(predictions=norm_pred, references=norm_ref),
        'raw_wer': 100 * wer_metric.compute(
            predictions=[predictions[i] for i in keep],
            references=[references[i] for i in keep]),
        'n': len(keep),
    }


def evaluate_asr(model, processor, dataset, label, **kwargs):
    print(f'[{label}] 추론 시작 — {len(dataset)}개 샘플')
    predictions = transcribe(model, processor, dataset, **kwargs)
    scores = compute_scores(predictions, dataset['sentence'])
    scores['predictions'] = predictions
    scores['label'] = label
    print(f"[{label}] WER {scores['wer']:.2f}% | CER {scores['cer']:.2f}%")
    return scores


def _width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in text)


def _pad(text, width):
    return text + ' ' * max(0, width - _width(text))


def print_comparison(base_scores, tuned_scores, label_width=16):
    header = _pad('모델', label_width) + f"{'WER':>10}{'CER':>10}{'raw WER':>12}"
    rule = '-' * _width(header)

    print('\n' + header)
    print(rule)
    for scores in (base_scores, tuned_scores):
        print(_pad(scores['label'], label_width)
              + f"{scores['wer']:>9.2f}%{scores['cer']:>9.2f}%{scores['raw_wer']:>11.2f}%")
    print(rule)

    for name in ('wer', 'cer'):
        delta = base_scores[name] - tuned_scores[name]
        rel = delta / base_scores[name] * 100 if base_scores[name] else 0.0
        print(f'{name.upper()} 개선: {delta:+.2f}%p ({rel:+.2f}% 상대)')


def print_samples(dataset, base_scores, tuned_scores, limit=10):
    for i in range(min(limit, len(dataset))):
        print(f'\n샘플 {i + 1}')
        print(f"정답     : {dataset[i]['sentence']}")
        print(f"베이스   : {base_scores['predictions'][i]}")
        print(f"튜닝     : {tuned_scores['predictions'][i]}")
