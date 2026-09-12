import tempfile

import gradio as gr
import librosa
import numpy as np
import requests
import soundfile as sf
import whisper

PROMPT = '사용자 입력을 보고, 제어 명령을 생성해주세요. 오직 JSON 객체만 생성하세요.'

TARGET_SAMPLE_RATE = 16000


def make_alpaca_prompt(instruction, input_text):
    if input_text.strip():
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


def prepare_audio(audio, target_sample_rate=TARGET_SAMPLE_RATE):
    input_sample_rate, raw_audio = audio
    audio_data = raw_audio.astype(np.float32)

    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = audio_data / max_val

    if input_sample_rate != target_sample_rate:
        audio_data = librosa.resample(
            audio_data, orig_sr=input_sample_rate, target_sr=target_sample_rate)
    return audio_data, target_sample_rate


def transcribe(wav_path, whisper_model, processor, whisper_tokenizer):
    audio_loaded = whisper.load_audio(wav_path)
    audio_loaded = whisper.pad_or_trim(audio_loaded)

    device = next(whisper_model.parameters()).device
    inputs = processor(
        audio_loaded,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors='pt',
    ).to(device)

    input_features = inputs.input_features.to(whisper_model.dtype)
    forced_decoder_ids = whisper_tokenizer.get_decoder_prompt_ids(
        language='korean', task='transcribe')
    predicted_ids = whisper_model.generate(
        input_features=input_features,
        forced_decoder_ids=forced_decoder_ids,
        max_length=448,
    )
    return whisper_tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]


def generate_command(transcription, llm_model, llm_tokenizer, prompt=PROMPT):
    alpaca_prompt = make_alpaca_prompt(prompt, transcription)
    inputs = llm_tokenizer(alpaca_prompt, return_tensors='pt').to(llm_model.device)
    outputs = llm_model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        eos_token_id=llm_tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    return llm_tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def send_to_local_server(llm_output, server_url, timeout=5):
    payload = {'llm_output': llm_output.strip()}
    print('전송할 payload:', payload)
    try:
        response = requests.post(server_url, json=payload, timeout=timeout)
        if response.status_code == 200:
            print('로컬 서버 응답:', response.json())
        else:
            print(f'로컬 서버 에러: {response.status_code}')
            print(f'에러 응답: {response.text}')
    except Exception as e:
        print(f'로컬 서버 통신 에러: {e}')


def build_interface(whisper_model, processor, whisper_tokenizer,
                    llm_model, llm_tokenizer, server_url, prompt=PROMPT):

    def process_audio(audio):
        if audio is None:
            return '음성이 감지되지 않았습니다.', '응답 없음'
        try:
            audio_data, sample_rate = prepare_audio(audio)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sample_rate)

                transcription = transcribe(
                    temp_file.name, whisper_model, processor, whisper_tokenizer)
                print('Transcription:', transcription)

                llm_output = generate_command(
                    transcription, llm_model, llm_tokenizer, prompt)
                print('LLM Response:', llm_output)

                send_to_local_server(llm_output, server_url)
                return transcription, llm_output
        except Exception as e:
            print(f'오류 발생: {str(e)}')
            return f'오류 발생: {str(e)}', '응답 없음'

    return gr.Interface(
        fn=process_audio,
        inputs=gr.Audio(sources=['microphone'], type='numpy'),
        outputs=[gr.Textbox(label='Transcription'), gr.Textbox(label='LLM Response')],
        title='OpenAI Whisper ASR & LLM Pipeline',
        description='마이크에 말씀하시면 음성을 인식하여 RC 카 제어 명령으로 변환합니다.',
    )
