# IndexTTS API server

`api_server.py` nạp model một lần khi khởi động và mở một HTTP API để sinh giọng nói. Có kèm trang nghe thử ở `/`.

## Cài đặt & chạy trên server (Linux + GPU NVIDIA)

```bash
git pull
uv sync --extra webui                                # fastapi/uvicorn đi kèm gradio
uv tool install "huggingface-hub"
hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints   # chỉ cần làm 1 lần

uv run api_server.py --host 0.0.0.0 --port 8000 --fp16 --api_key "doi-key-nay"
```

- Trang nghe thử: `http://<ip-server>:8000/`
- Swagger (thử từng endpoint): `http://<ip-server>:8000/docs`
- Muốn chạy nền sau khi thoát SSH: dùng `tmux`/`screen`, hoặc
  `nohup uv run api_server.py --fp16 --api_key xxx > api.log 2>&1 &`

| Tham số | Ý nghĩa |
|---|---|
| `--host` / `--port` | Mặc định `0.0.0.0:8000` |
| `--fp16` | BF16 (bản 2.5): nhanh hơn, ít VRAM hơn. GPU dưới 10 GB thì tự bật |
| `--api_key` | Bắt client gửi `X-API-Key: <key>` hoặc `Authorization: Bearer <key>`. Cũng có thể đặt qua biến môi trường `INDEXTTS_API_KEY` |
| `--qwen_emo` | Ép nạp QwenEmotion (dùng cho `emo_mode=text`) khi GPU dưới 10 GB |
| `--version 2 --model_dir ./checkpoints_2` | Dùng IndexTTS-2 cũ |
| `--voices_dir` | Thư mục lưu giọng mẫu, mặc định `./voices` |
| `--max_text_chars` | Độ dài văn bản tối đa, mặc định 3000 |

Thư mục `checkpoints/` của bản 2.5 phải có: `config.yaml`, `gpt.pth`, `s2mel.pth`, `codec.pth`, `multilingual_zh_ja_yue_char_del.tiktoken`, `wav2vec2bert_stats.pt`.

Mỗi lần server chỉ sinh **một** request. Request đến sau sẽ phải chờ. Lần đầu dùng một giọng mới sẽ chậm hơn, các lần sau với cùng giọng sẽ dùng cache.

## Endpoint

### `GET /health`
Trạng thái server, thiết bị, VRAM, có đang bận không. Endpoint này không cần key.

### `GET /voices`
Liệt kê giọng mẫu có sẵn (`examples/` + giọng đã lưu), các ngôn ngữ, và thứ tự nhãn cảm xúc.

### `POST /voices` (multipart): lưu giọng mẫu để dùng lại
- `name`: chữ, số, `_`, `-` (tối đa 64 ký tự)
- `audio`: file `.wav/.mp3/.flac/.ogg/.m4a`, tối đa 20 MB. Nên dùng khoảng 5–15 giây, giọng sạch.

### `DELETE /voices/{name}`
Xoá giọng đã lưu.

### `POST /tts` (multipart form): trả về `audio/wav`

| Trường | Mặc định | Mô tả |
|---|---|---|
| `text` | (bắt buộc) | Văn bản cần đọc |
| `lang` | `ZH` | `ZH`, `EN`, `JA`, `AR`, `ES` (bản 2.5 **không** hỗ trợ tiếng Việt) |
| `voice` | | Tên giọng từ `GET /voices` |
| `spk_audio` | | **Hoặc** upload thẳng file giọng mẫu (ưu tiên hơn `voice`) |
| `duration_factor` | `1.0` | 0.5–2.0. Lớn hơn 1 thì đọc chậm hơn, nhỏ hơn 1 thì đọc nhanh hơn |
| `emo_mode` | `speaker` | `speaker` (theo giọng mẫu), `audio`, `vector`, `text` |
| `emo_audio` / `emo_voice` | | Audio cảm xúc, dùng khi `emo_mode=audio` |
| `emo_vector` | | 8 số 0–1 theo thứ tự `happy,angry,sad,afraid,disgusted,melancholic,surprised,calm`, dùng khi `emo_mode=vector` |
| `emo_text` | | Mô tả cảm xúc, dùng khi `emo_mode=text`. Để trống thì lấy chính `text` |
| `emo_alpha` | `1.0` | Cường độ cảm xúc 0–1 |
| `emo_random` | `false` | Lấy mẫu cảm xúc ngẫu nhiên |
| Nâng cao | | `do_sample=true, top_p=0.8, top_k=30, temperature=0.8, num_beams=3, repetition_penalty=10, length_penalty=0, max_mel_tokens=1500, max_text_tokens_per_segment=120, interval_silence=200` |

Header trả về: `X-Inference-Seconds`, `X-Audio-Seconds`, `X-RTF`.
Khi có lỗi, server trả JSON `{"detail": "..."}` với mã 400/401/404/413/500.

## Ví dụ gọi

```bash
# Dùng giọng có sẵn
curl -X POST http://SERVER:8000/tts -H "X-API-Key: doi-key-nay" \
  -F text="Hello, nice to meet you." -F lang=EN -F voice=voice_01 -o out.wav

# Upload giọng của mình + cảm xúc vui
curl -X POST http://SERVER:8000/tts -H "X-API-Key: doi-key-nay" \
  -F text="Today is a great day!" -F lang=EN -F spk_audio=@my_voice.wav \
  -F emo_mode=vector -F emo_vector="0.8,0,0,0,0,0,0,0" -o happy.wav

# Lưu giọng để lần sau gọi bằng tên
curl -X POST http://SERVER:8000/voices -H "X-API-Key: doi-key-nay" -F name=minh -F audio=@my_voice.wav
```

```python
import requests
r = requests.post(
    "http://SERVER:8000/tts",
    headers={"X-API-Key": "doi-key-nay"},
    data={"text": "Hello world", "lang": "EN", "voice": "minh"},
    timeout=600,
)
r.raise_for_status()
open("out.wav", "wb").write(r.content)
```
