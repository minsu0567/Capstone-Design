import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset
from peft import PeftConfig, PeftModel, PeftType
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

JSON_ENCODINGS = ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr')

DEFAULT_TRAINING_ARGS = dict(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    warmup_steps=100,
    num_train_epochs=20,
    weight_decay=0.01,
    optim='adamw_8bit',
    lr_scheduler_type='constant_with_warmup',
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': True},
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_steps=10,
    report_to=['tensorboard'],
    load_best_model_at_end=True,
    metric_for_best_model='wer',
    greater_is_better=False,
    push_to_hub=False,
)


def load_json_dataset(json_path):
    for encoding in JSON_ENCODINGS:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        print(f'JSON 파일을 {encoding} 인코딩으로 성공적으로 로드했습니다.')
        return Dataset.from_list(data)
    raise ValueError(f'JSON 파일을 읽을 수 없습니다: {json_path}')


def combine_json_files(dialect_json_path, task_specific_json_path, output_json_path):
    with open(dialect_json_path, 'r', encoding='utf-8') as f:
        dialect_data = json.load(f)
    with open(task_specific_json_path, 'r', encoding='utf-8') as f:
        task_specific_data = json.load(f)

    combined_data = dialect_data + task_specific_data
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    return combined_data


def split_dataset_with_task_focus(dataset, task_marker='whisper_training3',
                                  task_specific_ratio=0.2, test_size=0.1):
    task_specific_data, general_data = [], []
    for item in dataset:
        if task_marker in (item.get('source') or item['audio']['path']):
            task_specific_data.append(item)
        else:
            general_data.append(item)

    task_test_size = int(len(task_specific_data) * task_specific_ratio)
    general_test_size = int(len(general_data) * test_size)

    train_data = task_specific_data[task_test_size:] + general_data[general_test_size:]
    test_data = task_specific_data[:task_test_size] + general_data[:general_test_size]
    return Dataset.from_list(train_data), Dataset.from_list(test_data)


def update_audio_paths(dataset, audio_base_path):
    def update_path(example):
        if 'audio' in example and 'path' in example['audio']:
            file_name = os.path.basename(example['audio']['path'])
            example['audio']['path'] = os.path.join(audio_base_path, file_name)
        return example

    return dataset.map(update_path)


class WhisperTuner(PeftModel):
    def __init__(self, model: torch.nn.Module, peft_config: PeftConfig,
                 adapter_name: str = 'default') -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )
        self.base_model_prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model._prepare_encoder_decoder_kwargs_for_generation
        )

    def forward(
            self,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            decoder_inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task_ids=None,
            **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs['task_ids'] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items()
                          if k not in self.special_peft_forward_args}
                return self.base_model(
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )


def prepare_dataset(batch, processor):
    audio = batch['audio']
    input_features = processor.feature_extractor(
        audio['array'],
        sampling_rate=audio['sampling_rate'],
    ).input_features[0]
    labels = processor.tokenizer(batch['sentence']).input_ids
    return {'input_features': input_features, 'labels': labels}


def preprocess_datasets(train_dataset, test_dataset, processor,
                        sampling_rate=16000, batch_size=16):
    processed = []
    for name, dataset in (('Train', train_dataset), ('Test', test_dataset)):
        print(f'{name} 데이터셋 전처리 중...')
        dataset = dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))
        dataset = dataset.map(
            prepare_dataset,
            fn_kwargs={'processor': processor},
            remove_columns=dataset.column_names,
            num_proc=None,
            batch_size=batch_size,
        )
        processed.append(dataset)
    return processed[0], processed[1]


def validate_dataset(dataset, name='Dataset'):
    try:
        assert 'input_features' in dataset.features, f'{name}에 input_features가 없습니다.'
        assert 'labels' in dataset.features, f'{name}에 labels가 없습니다.'

        sample = dataset[0]
        assert isinstance(sample['input_features'], (list, np.ndarray)), \
            f'{name}의 input_features 타입이 잘못되었습니다.'
        assert isinstance(sample['labels'], list), \
            f'{name}의 labels 타입이 잘못되었습니다.'

        print(f'{name} 검증 완료! '
              f'input_features={np.asarray(sample["input_features"]).shape} '
              f'labels={len(sample["labels"])}')
        return True
    except Exception as e:
        print(f'{name} 검증 실패: {str(e)}')
        return False


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
                 ) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': f['input_features']} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': f['labels']} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        labels = labels_batch['input_ids'].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels
        return batch


def build_compute_metrics(tokenizer):
    metric = evaluate.load('wer')

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {'wer': 100 * metric.compute(predictions=pred_str, references=label_str)}

    return compute_metrics


def build_training_args(output_dir, **overrides):
    cfg = dict(DEFAULT_TRAINING_ARGS)
    cfg.update(overrides)
    cfg['output_dir'] = output_dir

    fields = set(Seq2SeqTrainingArguments.__dataclass_fields__)
    if 'eval_strategy' in cfg and 'eval_strategy' not in fields:
        cfg['evaluation_strategy'] = cfg.pop('eval_strategy')

    dropped = [k for k in cfg if k not in fields]
    for k in dropped:
        cfg.pop(k)
    if dropped:
        print('[build_training_args] not supported by this transformers version, '
              'dropped:', dropped)
    return Seq2SeqTrainingArguments(**cfg)


def build_trainer(model, processor, tokenizer, train_dataset, eval_dataset,
                  output_dir, **overrides):
    import inspect

    training_args = build_training_args(output_dir, **overrides)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer_params = set(inspect.signature(Seq2SeqTrainer.__init__).parameters)
    processing = ({'processing_class': processor} if 'processing_class' in trainer_params
                  else {'tokenizer': processor})

    return Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        **processing,
    )
