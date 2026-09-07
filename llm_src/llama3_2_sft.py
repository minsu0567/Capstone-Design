from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

ALPACA_HEADER = ('Below is an instruction that describes a task. '
                 'Write a response that appropriately completes the request.')

DEFAULT_SFT_CONFIG = dict(
    report_to='tensorboard',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    warmup_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    logging_steps=10,
    optim='adamw_8bit',
    weight_decay=0.01,
    lr_scheduler_type='constant_with_warmup',
    seed=42,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs={'use_reentrant': True},
    max_length=128,
    dataset_num_proc=2,
    packing=False,
    completion_only_loss=True,
)


def make_alpaca_prompt(instruction, input_text):
    if input_text.strip():
        return f"""{ALPACA_HEADER}

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""{ALPACA_HEADER}

### Instruction:
{instruction}

### Response:
"""


def build_formatting_func(tokenizer):
    eos_token = tokenizer.eos_token

    def prompt_formatting_func(examples):
        prompts, completions = [], []
        for instruction, input_text, output in zip(
                examples['instruct'], examples['input'], examples['output']):
            prompts.append(make_alpaca_prompt(instruction, input_text))
            completions.append(str(output) + eos_token)
        return {'prompt': prompts, 'completion': completions}

    return prompt_formatting_func


def build_dataset(data_files, tokenizer, test_size=0.01, seed=42):
    dataset = load_dataset('json', data_files=data_files, split='train')
    dataset = dataset.shuffle(seed=seed)
    mapped = dataset.map(
        build_formatting_func(tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )
    split = mapped.train_test_split(test_size=test_size, seed=seed)
    return split['train'], split['test']


def build_sft_config(output_dir, **overrides):
    cfg = dict(DEFAULT_SFT_CONFIG)
    cfg.update(overrides)
    cfg['output_dir'] = output_dir

    fields = set(SFTConfig.__dataclass_fields__)
    dropped = [k for k in cfg if k not in fields]
    for k in dropped:
        cfg.pop(k)
    if dropped:
        print('[build_sft_config] not supported by this trl version, dropped:', dropped)
    return SFTConfig(**cfg)


def build_trainer(model, tokenizer, train_dataset, eval_dataset, peft_config,
                  output_dir, **overrides):
    args = build_sft_config(output_dir, **overrides)
    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        args=args,
    )


def generate_response(prompt, model, tokenizer, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_model(test_dataset, instruction, model, tokenizer,
                   num_samples=None, max_new_tokens=128, log_every=5):
    if num_samples is None:
        num_samples = len(test_dataset)

    results = []
    correct = 0
    for i, example in enumerate(test_dataset):
        if i >= num_samples:
            break

        full_prompt = make_alpaca_prompt(instruction, example['input'])
        model_response = generate_response(
            full_prompt, model, tokenizer, max_new_tokens=max_new_tokens)

        expected_output = str(example['output'])
        model_output = model_response.split('### Response:')[-1].strip()
        is_correct = expected_output == model_output

        results.append({
            'input': example['input'],
            'expected_output': expected_output,
            'model_output': model_output,
            'is_correct': is_correct,
        })
        correct += int(is_correct)

        if log_every and (i + 1) % log_every == 0:
            print(f'평가 진행 중: {i + 1}/{num_samples} 완료')

    return results, correct / num_samples


def print_results(results, limit=None):
    for i, result in enumerate(results[:limit]):
        print(f"\n샘플 {i + 1}:")
        print(f"입력      : {result['input']}")
        print(f"기대 출력 : {result['expected_output']}")
        print(f"모델 출력 : {result['model_output']}")
        print(f"정답 여부 : {result['is_correct']}")
