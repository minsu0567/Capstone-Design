import json
import random
from collections import Counter, defaultdict

import command_lexicon as lex
import generate_multi_commands as multi
import generate_single_commands as single

TRAIN_FILE = "total_commands.jsonl"
SIMPLE_FILE = "simple_commands.jsonl"
COMPLEX_FILE = "complex_commands.jsonl"
TEST_FILE = "test.jsonl"

TEST_SIZE = 100
SEED = 42

# generate_single_commands.py가 문구당 생성하는 샘플 수
SINGLE_REPEATS = 9

# 복합 명령의 마지막 자리는 stop으로 고정하므로 앞자리 후보에서 제외한다.
LEAD_COMMAND_IDS = [c for c in multi.MULTI_COMMAND_IDS if c != "stop"]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def target_counts(weights, total, minimum=1):
    """가중치에 비례하도록 total을 배분하되 각 항목에 최소 minimum을 보장한다."""
    keys = list(weights)
    weight_sum = sum(weights.values())
    exact = {k: weights[k] / weight_sum * total for k in keys}

    counts = {k: max(minimum, int(exact[k])) for k in keys}
    remainder = sorted(keys, key=lambda k: exact[k] - int(exact[k]), reverse=True)

    i = 0
    while sum(counts.values()) < total:
        counts[remainder[i % len(remainder)]] += 1
        i += 1
    while sum(counts.values()) > total:
        k = max(keys, key=lambda k: counts[k] - exact[k])
        if counts[k] <= minimum:
            break
        counts[k] -= 1
    return counts


def build_single_pool():
    pool = defaultdict(list)
    for example in single.generate_single_command_examples(num_samples_per_phrase=1):
        pool[example["output"]["commands"][0]["command_id"]].append(example)
    return pool


def training_length_weights(pool):
    """학습셋의 명령 개수별 비율. 이미 홀드아웃이 빠진 파일을 읽으면 비율이 흔들리므로
    생성기 원본 기준으로 계산해 재실행해도 같은 결과가 나오게 한다."""
    weights = {1: sum(len(v) for v in pool.values()) * SINGLE_REPEATS}
    for item in load_jsonl(COMPLEX_FILE):
        n = len(item["output"]["commands"])
        weights[n] = weights.get(n, 0) + 1
    return weights


def pick_single_examples(pool, count, rng):
    """단일 명령은 어휘 공간이 학습셋에 모두 포함되어 있어 문구 단위로 홀드아웃한다."""
    counts = target_counts({cid: len(items) for cid, items in pool.items()}, count)

    picked = []
    for command_id, n in counts.items():
        candidates = sorted(pool[command_id], key=lambda e: e["input"])
        picked.extend(rng.sample(candidates, min(n, len(candidates))))
    return picked


def pick_multi_examples(train_inputs, counts, rng, max_attempts=200000):
    """복합 명령은 조합 공간이 넓어 학습셋에 없는 입력만 새로 뽑는다.
    마지막 명령은 항상 stop이다."""
    picked = {}
    attempts = 0

    def filled(n):
        return sum(1 for e in picked.values() if len(e["output"]["commands"]) == n)

    while sum(counts.values()) > len(picked) and attempts < max_attempts:
        attempts += 1
        n = rng.choice([k for k, v in counts.items() if filled(k) < v])
        combo = rng.sample(LEAD_COMMAND_IDS, n - 1) + ["stop"]

        phrases, commands = [], []
        for i, command_id in enumerate(combo):
            is_last = (i == len(combo) - 1)
            pattern = rng.choice(multi.COMMAND_PATTERNS[command_id])
            duration = -1 if is_last else rng.choice(lex.DURATIONS)
            phrases.append(multi.make_phrase(command_id, pattern, duration, is_last))
            commands.append({"command_id": command_id, "value": -1, "duration": duration})

        text = " ".join(phrases)
        if text in train_inputs or text in picked:
            continue
        picked[text] = lex.make_example(text, commands)

    if len(picked) < sum(counts.values()):
        raise RuntimeError(
            f"복합 명령 후보가 부족합니다: {len(picked)}/{sum(counts.values())}")
    return list(picked.values())


def main():
    rng = random.Random(SEED)
    random.seed(SEED)

    pool = build_single_pool()
    length_counts = target_counts(training_length_weights(pool), TEST_SIZE, minimum=0)
    print("길이별 목표 개수:", dict(sorted(length_counts.items())))

    train_inputs = {item["input"] for item in load_jsonl(TRAIN_FILE)}

    singles = pick_single_examples(pool, length_counts.pop(1, 0), rng)
    multis = pick_multi_examples(train_inputs, length_counts, rng)

    test = singles + multis
    rng.shuffle(test)
    lex.save_to_jsonl(test, TEST_FILE)

    held_out = {item["input"] for item in test}
    for path in (TRAIN_FILE, SIMPLE_FILE):
        rows = load_jsonl(path)
        kept = [item for item in rows if item["input"] not in held_out]
        lex.save_to_jsonl(kept, path)
        print(f"  {path}: {len(rows)} -> {len(kept)} (제거 {len(rows) - len(kept)})")

    print("테스트셋 길이 분포:",
          dict(sorted(Counter(len(i['output']['commands']) for i in test).items())))
    print("마지막 명령이 stop이 아닌 복합 명령:",
          sum(1 for i in test
              if len(i['output']['commands']) > 1
              and i['output']['commands'][-1]['command_id'] != 'stop'))
    print("테스트셋 명령 분포:",
          dict(Counter(c['command_id'] for i in test for c in i['output']['commands'])))


if __name__ == "__main__":
    main()
