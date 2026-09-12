import random

import command_lexicon as lex

OUTPUT_FILE = "simple_commands.jsonl"


def _with_endings(phrase, style):
    for ending in lex.ENDINGS[style]:
        yield f"{phrase} {ending}"


def single_phrases():
    """(command_id, 명령 문장) 조합을 모두 생성한다."""
    # 전진 / 후진
    for command_id, phrases in lex.MOVE_PHRASES.items():
        for phrase in phrases:
            yield from ((command_id, s) for s in _with_endings(phrase, lex.move_style(phrase)))

    # 좌 / 우회전: 방향구는 회전하다·돌리다, 기본 표현은 하다 어미
    for command_id, phrases in lex.DIRECTION_PHRASES.items():
        for phrase in phrases:
            for style in ("회전하다", "돌리다"):
                yield from ((command_id, s) for s in _with_endings(phrase, style))
        yield from ((command_id, s)
                    for s in _with_endings(lex.TURN_KEYWORDS[command_id], "하다"))

    # 후진 좌/우회전
    for command_id, phrases in lex.BACKWARD_TURN_PHRASES.items():
        for phrase in phrases:
            yield from ((command_id, s) for s in _with_endings(phrase, "하다"))

    # 회전 속도 (빨리 / 천천히)
    for command_id, phrases in lex.TURN_SPEED_PHRASES.items():
        for phrase in phrases:
            for style in ("회전하다", "돌리다"):
                yield from ((command_id, s) for s in _with_endings(phrase, style))

    # 속도 증감: 어미가 이미 포함된 완성형
    for command_id, phrases in lex.SPEED_STEP_PHRASES.items():
        for phrase in phrases:
            yield command_id, phrase

    # 정지
    for phrase in lex.stop_phrases():
        yield "stop", phrase

    # 종료
    for phrase in lex.QUIT_PHRASES:
        yield from (("quit", s) for s in _with_endings(phrase, "하다"))


def speed_setting_examples(num_samples_per_phrase=1):
    """속도 설정 명령: 값이 value 필드로 들어가는 유일한 명령."""
    results = []
    for particle, values in lex.SPEED_VALUES.items():
        for value in values:
            for pattern in lex.SPEED_PATTERNS[particle]:
                for _ in range(num_samples_per_phrase):
                    results.append(lex.make_example(
                        pattern.format(value),
                        [{"command_id": "setting",
                          "value": lex.korean_to_number(value)}],
                    ))
    return results


def generate_single_command_examples(num_samples_per_phrase=1):
    results = []
    for command_id, phrase in single_phrases():
        for _ in range(num_samples_per_phrase):
            results.append(lex.make_example(
                phrase,
                [{"command_id": command_id, "value": -1}],
            ))
    results.extend(speed_setting_examples(num_samples_per_phrase))
    return results


if __name__ == "__main__":
    single_cmds = generate_single_command_examples(num_samples_per_phrase=9)
    random.shuffle(single_cmds)
    lex.save_to_jsonl(single_cmds, OUTPUT_FILE)
