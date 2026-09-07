import itertools
import random

import command_lexicon as lex

OUTPUT_FILE = "complex_commands.jsonl"

MULTI_COMMAND_IDS = ["forward", "backward", "right", "left", "stop"]


def _multi_patterns(command_id):
    """복합 명령에서 쓰는 표현 목록 (어미가 붙기 전 형태)."""
    if command_id in lex.MOVE_PHRASES:
        return lex.MOVE_PHRASES[command_id]
    if command_id in lex.DIRECTION_PHRASES:
        directions = lex.DIRECTION_PHRASES[command_id]
        return ([lex.TURN_KEYWORDS[command_id]] + directions
                + [f"{d} 회전" for d in directions])
    if command_id == "stop":
        return lex.stop_phrases()
    raise ValueError(f"복합 명령에서 지원하지 않는 명령: {command_id}")


COMMAND_PATTERNS = {cmd: _multi_patterns(cmd) for cmd in MULTI_COMMAND_IDS}


def _ending_style(command_id, pattern):
    if command_id in lex.MOVE_PHRASES:
        return lex.move_style(pattern)
    if command_id in lex.DIRECTION_PHRASES:
        # "회전"이 들어간 표현은 하다 계열, 방향구만 있으면 돌리다 계열
        return "하다" if "회전" in pattern else "돌리다"
    return "하다"


def make_phrase(command_id, pattern, duration, is_last):
    """명령 하나에 대한 한국어 표현을 만든다."""
    if command_id == "stop":
        if is_last:
            return pattern
        return f"{lex.DURATION_KOR[duration]}초간 {random.choice(lex.CONNECTIVES['멈추다'])}"

    style = _ending_style(command_id, pattern)
    if is_last:
        return f"{pattern} {random.choice(lex.ENDINGS[style])}"
    return f"{lex.DURATION_KOR[duration]}초간 {pattern} {random.choice(lex.CONNECTIVES[style])}"


def generate_multi_command_examples(num_samples_per_combo=2):
    results = []
    for n in (2, 3):
        for combo in itertools.permutations(MULTI_COMMAND_IDS, n):
            if combo[0] == "stop":
                continue
            for _ in range(num_samples_per_combo):
                phrases = []
                commands = []
                for i, command_id in enumerate(combo):
                    is_last = (i == len(combo) - 1)
                    pattern = random.choice(COMMAND_PATTERNS[command_id])
                    # 마지막 명령은 지속시간 없이 종결 어미로 끝낸다
                    duration = -1 if is_last else random.choice(lex.DURATIONS)
                    phrases.append(make_phrase(command_id, pattern, duration, is_last))
                    commands.append({"command_id": command_id,
                                     "value": -1,
                                     "duration": duration})
                results.append(lex.make_example(" ".join(phrases), commands))
    return results


if __name__ == "__main__":
    multi_cmds = generate_multi_command_examples(num_samples_per_combo=300)
    random.shuffle(multi_cmds)
    lex.save_to_jsonl(multi_cmds, OUTPUT_FILE)
