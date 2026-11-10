"""
IndexTTS HTTP API server.

Loads the model once at startup and exposes a small REST API (plus a tiny
browser test page at "/") for generating speech.

Run (on the GPU server):
    uv sync --extra webui            # fastapi/uvicorn/python-multipart come with gradio
    uv run api_server.py --host 0.0.0.0 --port 8000 --fp16

See docs/api_server_vi.md for the full endpoint reference.
"""

import argparse
import hashlib
import os
import re
import sys
import threading
import time
import uuid
import wave

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

parser = argparse.ArgumentParser(description="IndexTTS HTTP API server")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
parser.add_argument("--port", type=int, default=8000, help="Port to bind")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--version", type=str, default="2.5", choices=["2", "2.5"], help="Model version to use")
parser.add_argument("--fp16", action="store_true", default=False, help="Half precision (BF16 on 2.5, FP16 on 2)")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use BigVGAN CUDA kernel if available")
parser.add_argument("--accel", action="store_true", default=False, help="Use GPT2 acceleration engine if available")
parser.add_argument("--torch_compile", action="store_true", default=False, help="Use torch.compile for s2mel")
parser.add_argument("--qwen_emo", action="store_true", default=False,
                    help="Load QwenEmotion even on a low-VRAM GPU, where it is skipped by default")
parser.add_argument("--api_key", type=str, default=os.environ.get("INDEXTTS_API_KEY"),
                    help="If set, requests must send 'X-API-Key: <key>' or 'Authorization: Bearer <key>' "
                         "(default: $INDEXTTS_API_KEY)")
parser.add_argument("--voices_dir", type=str, default="./voices", help="Where saved reference voices live")
parser.add_argument("--max_text_chars", type=int, default=3000, help="Reject texts longer than this")
parser.add_argument("--verbose", action="store_true", default=False, help="Verbose inference logs")
cmd_args = parser.parse_args()

IS_V25 = cmd_args.version == "2.5"

if IS_V25:
    required = ["gpt.pth", "s2mel.pth", "codec.pth", "multilingual_zh_ja_yue_char_del.tiktoken",
                "wav2vec2bert_stats.pt", "config.yaml"]
else:
    required = ["bpe.model", "gpt.pth", "s2mel.pth", "wav2vec2bert_stats.pt", "config.yaml"]
missing = [f for f in required if not os.path.exists(os.path.join(cmd_args.model_dir, f))]
if missing:
    print(f">> Missing model files in {cmd_args.model_dir}: {missing}")
    print(">> Download them first, e.g.: hf download IndexTeam/IndexTTS-2.5 --local-dir=checkpoints")
    sys.exit(1)

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402

if IS_V25:
    from indextts.infer_v2_5 import IndexTTS2
else:
    from indextts.infer_v2 import IndexTTS2

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

LOW_VRAM_THRESHOLD_GB = 10.0
_vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if torch.cuda.is_available() else None)
LOW_VRAM = _vram_gb is not None and _vram_gb < LOW_VRAM_THRESHOLD_GB
HALF_PRECISION = cmd_args.fp16 or LOW_VRAM
LOAD_QWEN_EMO = cmd_args.qwen_emo or not LOW_VRAM

if _vram_gb is None:
    print(">> WARNING: CUDA is not available, inference will run on CPU and be very slow.")
elif LOW_VRAM:
    print(f">> {_vram_gb:.1f} GB VRAM detected (< {LOW_VRAM_THRESHOLD_GB:.0f} GB): enabling half precision"
          + ("" if LOAD_QWEN_EMO else " and skipping QwenEmotion (emo_mode=text unavailable)"))

tts_kwargs = dict(
    model_dir=cmd_args.model_dir,
    cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
    use_deepspeed=cmd_args.deepspeed,
    use_cuda_kernel=cmd_args.cuda_kernel,
    use_accel=cmd_args.accel,
    use_torch_compile=cmd_args.torch_compile,
    use_qwen_emo=LOAD_QWEN_EMO,
)
if IS_V25:
    use_bf16 = HALF_PRECISION and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if HALF_PRECISION and not use_bf16:
        print(">> BF16 is not supported on this device, falling back to full precision.")
    tts_kwargs["use_bf16"] = use_bf16
else:
    tts_kwargs["use_fp16"] = HALF_PRECISION

print(">> Loading model...")
tts = IndexTTS2(**tts_kwargs)
print(">> Model loaded.")

try:
    from indextts.utils.examples_downloader import ensure_examples_available
    ensure_examples_available()
except Exception as e:  # examples are only a convenience; never block startup on them
    print(f">> Could not download example voices: {e}")

# The model holds GPU state and a speaker cache, so only one generation runs at a time.
INFER_LOCK = threading.Lock()

# --------------------------------------------------------------------------- #
# Voices / uploads
# --------------------------------------------------------------------------- #

LANGUAGES = ["ZH", "EN", "JA", "AR", "ES"]
EMO_LABELS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

VOICES_DIR = os.path.abspath(cmd_args.voices_dir)
EXAMPLES_DIR = os.path.join(current_dir, "examples")
UPLOAD_CACHE_DIR = os.path.join(current_dir, "cache", "api_uploads")
OUTPUT_DIR = os.path.join(current_dir, "outputs", "api")
for d in (VOICES_DIR, UPLOAD_CACHE_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)


def _audio_files(directory):
    if not os.path.isdir(directory):
        return {}
    out = {}
    for fn in sorted(os.listdir(directory)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in AUDIO_EXTS and VOICE_NAME_RE.match(stem):
            out.setdefault(stem, os.path.join(directory, fn))
    return out


def list_voices():
    """Saved voices take precedence over bundled examples with the same name."""
    voices = {name: ("example", path) for name, path in _audio_files(EXAMPLES_DIR).items()}
    voices.update({name: ("saved", path) for name, path in _audio_files(VOICES_DIR).items()})
    return voices


def resolve_voice(name):
    if not VOICE_NAME_RE.match(name or ""):
        raise HTTPException(400, f"Invalid voice name: {name!r}")
    entry = list_voices().get(name)
    if entry is None:
        raise HTTPException(404, f"Voice not found: {name!r}. See GET /voices.")
    return entry[1]


def _check_ext(upload: UploadFile):
    ext = os.path.splitext(upload.filename or "")[1].lower() or ".wav"
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, f"Unsupported audio type {ext!r}; allowed: {sorted(AUDIO_EXTS)}")
    return ext


def _read_upload(upload: UploadFile):
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Audio file too large (max 20 MB)")
    if not data:
        raise HTTPException(400, "Empty audio file")
    return data


def save_upload_to_cache(upload: UploadFile):
    """Store by content hash so re-sending the same file hits the model's speaker cache."""
    ext = _check_ext(upload)
    data = _read_upload(upload)
    path = os.path.join(UPLOAD_CACHE_DIR, hashlib.sha256(data).hexdigest() + ext)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


def wav_duration_seconds(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title=f"IndexTTS-{cmd_args.version} API", version="1.0")


def require_api_key(request: Request):
    if not cmd_args.api_key:
        return
    key = request.headers.get("x-api-key")
    auth = request.headers.get("authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    if key != cmd_args.api_key:
        raise HTTPException(401, "Invalid or missing API key")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": cmd_args.version,
        "device": str(getattr(tts, "device", "unknown")),
        "vram_gb": round(_vram_gb, 1) if _vram_gb else None,
        "emo_text_available": getattr(tts, "qwen_emo", None) is not None,
        "busy": INFER_LOCK.locked(),
        "auth_required": bool(cmd_args.api_key),
    }


@app.get("/voices", dependencies=[Depends(require_api_key)])
def get_voices():
    return {
        "voices": [{"name": n, "source": src} for n, (src, _) in sorted(list_voices().items())],
        "languages": LANGUAGES if IS_V25 else [],
        "emotions": EMO_LABELS,
    }


@app.post("/voices", dependencies=[Depends(require_api_key)])
def add_voice(name: str = Form(...), audio: UploadFile = File(...)):
    if not VOICE_NAME_RE.match(name):
        raise HTTPException(400, "name must be 1-64 chars of letters, digits, '_' or '-'")
    ext = _check_ext(upload=audio)
    data = _read_upload(audio)
    for old in _audio_files(VOICES_DIR).values():
        if os.path.splitext(os.path.basename(old))[0] == name:
            os.remove(old)
    path = os.path.join(VOICES_DIR, name + ext)
    with open(path, "wb") as f:
        f.write(data)
    return {"name": name, "source": "saved"}


@app.delete("/voices/{name}", dependencies=[Depends(require_api_key)])
def delete_voice(name: str):
    if not VOICE_NAME_RE.match(name):
        raise HTTPException(400, "Invalid voice name")
    path = _audio_files(VOICES_DIR).get(name)
    if path is None:
        raise HTTPException(404, "Saved voice not found (bundled examples cannot be deleted)")
    os.remove(path)
    return {"deleted": name}


def _parse_emo_vector(raw):
    try:
        vals = [float(v) for v in re.split(r"[,\s]+", raw.strip().strip("[]")) if v]
    except ValueError:
        raise HTTPException(400, "emo_vector must be 8 numbers, e.g. '0,0,0,0,0,0,0.5,0'")
    if len(vals) != 8 or any(v < 0 or v > 1 for v in vals):
        raise HTTPException(400, f"emo_vector must be 8 numbers in [0,1] ordered {EMO_LABELS}")
    return vals


@app.post("/tts", dependencies=[Depends(require_api_key)])
def synthesize(
    text: str = Form(...),
    lang: str = Form("ZH"),
    voice: str = Form(None),
    spk_audio: UploadFile = File(None),
    emo_mode: str = Form("speaker"),
    emo_audio: UploadFile = File(None),
    emo_voice: str = Form(None),
    emo_alpha: float = Form(1.0),
    emo_vector: str = Form(None),
    emo_text: str = Form(None),
    emo_random: bool = Form(False),
    duration_factor: float = Form(1.0),
    max_text_tokens_per_segment: int = Form(120),
    interval_silence: int = Form(200),
    do_sample: bool = Form(True),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    temperature: float = Form(0.8),
    length_penalty: float = Form(0.0),
    num_beams: int = Form(3),
    repetition_penalty: float = Form(10.0),
    max_mel_tokens: int = Form(1500),
):
    """Generate speech. Returns audio/wav. See docs/api_server_vi.md for parameters."""
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "text is empty")
    if len(text) > cmd_args.max_text_chars:
        raise HTTPException(400, f"text too long ({len(text)} > {cmd_args.max_text_chars} chars)")

    lang = (lang or "ZH").upper()
    if IS_V25 and lang not in LANGUAGES:
        raise HTTPException(400, f"lang must be one of {LANGUAGES}")
    if not 0.5 <= duration_factor <= 2.0:
        raise HTTPException(400, "duration_factor must be in [0.5, 2.0]")
    if not 0.0 <= emo_alpha <= 1.0:
        raise HTTPException(400, "emo_alpha must be in [0, 1]")
    if not 20 <= max_text_tokens_per_segment <= 600:
        raise HTTPException(400, "max_text_tokens_per_segment must be in [20, 600]")

    # Speaker reference: uploaded file wins, else named voice.
    if spk_audio is not None and spk_audio.filename:
        spk_path = save_upload_to_cache(spk_audio)
    elif voice:
        spk_path = resolve_voice(voice)
    else:
        raise HTTPException(400, "Provide a reference voice: 'voice' (name from GET /voices) or 'spk_audio' (file)")

    emo_mode = (emo_mode or "speaker").lower()
    emo_path, vec, use_emo_text = None, None, False
    if emo_mode == "speaker":
        pass
    elif emo_mode == "audio":
        if emo_audio is not None and emo_audio.filename:
            emo_path = save_upload_to_cache(emo_audio)
        elif emo_voice:
            emo_path = resolve_voice(emo_voice)
        else:
            raise HTTPException(400, "emo_mode=audio needs 'emo_audio' (file) or 'emo_voice' (name)")
    elif emo_mode == "vector":
        if not emo_vector:
            raise HTTPException(400, "emo_mode=vector needs 'emo_vector' (8 numbers)")
        vec = tts.normalize_emo_vec(_parse_emo_vector(emo_vector), apply_bias=True)
    elif emo_mode == "text":
        if getattr(tts, "qwen_emo", None) is None:
            raise HTTPException(400, "emo_mode=text unavailable: QwenEmotion not loaded (start with --qwen_emo)")
        use_emo_text = True
    else:
        raise HTTPException(400, "emo_mode must be one of: speaker, audio, vector, text")

    output_path = os.path.join(OUTPUT_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav")
    infer_kwargs = dict(
        spk_audio_prompt=spk_path,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_path,
        emo_alpha=float(emo_alpha),
        emo_vector=vec,
        use_emo_text=use_emo_text,
        emo_text=(emo_text or None),
        use_random=bool(emo_random),
        interval_silence=int(interval_silence),
        verbose=cmd_args.verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        do_sample=bool(do_sample),
        top_p=float(top_p),
        top_k=int(top_k) if int(top_k) > 0 else None,
        temperature=float(temperature),
        length_penalty=float(length_penalty),
        num_beams=int(num_beams),
        repetition_penalty=float(repetition_penalty),
        max_mel_tokens=int(max_mel_tokens),
    )
    if IS_V25:
        infer_kwargs["lang"] = lang
        infer_kwargs["duration_factor"] = float(duration_factor)

    start = time.perf_counter()
    try:
        with INFER_LOCK:
            result = tts.infer(**infer_kwargs)
    except HTTPException:
        raise
    except Exception as e:
        print(f">> Inference failed: {e!r}")
        raise HTTPException(500, f"Inference failed: {e}")
    elapsed = time.perf_counter() - start

    if not result or not os.path.isfile(output_path):
        raise HTTPException(500, "Inference produced no audio")
    try:
        with open(output_path, "rb") as f:
            data = f.read()
        duration = wav_duration_seconds(output_path)
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass

    headers = {
        "X-Inference-Seconds": f"{elapsed:.2f}",
        "Content-Disposition": 'inline; filename="tts.wav"',
    }
    if duration:
        headers["X-Audio-Seconds"] = f"{duration:.2f}"
        headers["X-RTF"] = f"{elapsed / duration:.3f}"
    return Response(content=data, media_type="audio/wav", headers=headers)


@app.get("/", response_class=HTMLResponse)
def index():
    return TEST_PAGE


TEST_PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IndexTTS API</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--muted:#5f6b7a;--line:#d9dee5;--acc:#2563eb;--err:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--fg:#e6e9ee;--muted:#9aa4b2;--line:#2a2f3a;--acc:#6ea0ff;--err:#ff8a80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px}h1{font-size:20px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
textarea,input,select{width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);font:inherit}
textarea{min-height:110px;resize:vertical}.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.vec{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:4px 12px}.vec input{padding:0}
button{margin-top:14px;padding:10px 18px;border:0;border-radius:6px;background:var(--acc);color:#fff;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}.hide{display:none}#status{margin-top:10px;color:var(--muted)}#status.err{color:var(--err)}
audio{width:100%;margin-top:12px}a{color:var(--acc)}
</style></head><body><main>
<h1>IndexTTS API</h1><p class="sub">Trang nghe thử. Tài liệu API: <a href="/docs">/docs</a></p>
<div class="card">
  <label for="text">Văn bản</label><textarea id="text">Hello, this is a quick test of the IndexTTS API.</textarea>
  <div class="row">
    <div><label for="lang">Ngôn ngữ</label><select id="lang"></select></div>
    <div><label for="voice">Giọng mẫu</label><select id="voice"></select></div>
    <div><label for="dur">Hệ số thời lượng</label><input id="dur" type="number" min="0.5" max="2" step="0.05" value="1"></div>
  </div>
  <label for="spk">Hoặc tải file giọng mẫu lên (ghi đè lựa chọn ở trên)</label><input id="spk" type="file" accept="audio/*">
  <div class="row">
    <div><label for="emo">Cảm xúc</label><select id="emo">
      <option value="speaker">Theo giọng mẫu</option><option value="audio">Từ audio cảm xúc</option>
      <option value="vector">Theo vector</option><option value="text">Theo mô tả văn bản</option></select></div>
    <div><label for="alpha">Cường độ cảm xúc</label><input id="alpha" type="number" min="0" max="1" step="0.05" value="1"></div>
  </div>
  <div id="emoAudioBox" class="hide"><label for="emoAudio">File audio cảm xúc</label><input id="emoAudio" type="file" accept="audio/*"></div>
  <div id="emoVecBox" class="hide"><label>Vector cảm xúc (0–1)</label><div class="vec" id="vec"></div></div>
  <div id="emoTextBox" class="hide"><label for="emoText">Mô tả cảm xúc (để trống = dùng chính văn bản)</label><input id="emoText"></div>
  <label for="key">API key (nếu server có đặt)</label><input id="key" type="password" autocomplete="off">
  <button id="go">Tạo giọng nói</button><div id="status"></div>
  <audio id="player" controls class="hide"></audio><div id="dl"></div>
</div></main>
<script>
const $=id=>document.getElementById(id);
const store={get:k=>{try{return localStorage.getItem(k)}catch(e){return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
$('key').value=store.get('indextts_key')||'';
const hdr=()=>{const k=$('key').value.trim();return k?{'X-API-Key':k}:{}};
const labels={happy:'Vui',angry:'Giận',sad:'Buồn',afraid:'Sợ',disgusted:'Ghê tởm',melancholic:'U sầu',surprised:'Ngạc nhiên',calm:'Bình tĩnh'};
async function loadVoices(){
  const r=await fetch('/voices',{headers:hdr()});
  if(!r.ok){setStatus('Không tải được danh sách giọng: '+(await r.text()),true);return}
  const d=await r.json();
  $('voice').innerHTML=d.voices.map(v=>`<option value="${v.name}">${v.name}${v.source==='saved'?' (đã lưu)':''}</option>`).join('');
  $('lang').innerHTML=(d.languages.length?d.languages:['ZH']).map(l=>`<option${l==='EN'?' selected':''}>${l}</option>`).join('');
  $('vec').innerHTML=d.emotions.map((e,i)=>`<label>${labels[e]||e} <span id="v${i}o">0</span><input type="range" id="v${i}" min="0" max="1" step="0.05" value="0" oninput="$('v${i}o').textContent=this.value"></label>`).join('');
}
function setStatus(t,err){$('status').textContent=t;$('status').className=err?'err':''}
$('emo').onchange=()=>{const m=$('emo').value;$('emoAudioBox').classList.toggle('hide',m!=='audio');$('emoVecBox').classList.toggle('hide',m!=='vector');$('emoTextBox').classList.toggle('hide',m!=='text')};
$('key').onchange=()=>{store.set('indextts_key',$('key').value.trim());loadVoices()};
$('go').onclick=async()=>{
  const f=new FormData();f.append('text',$('text').value);f.append('lang',$('lang').value);
  f.append('duration_factor',$('dur').value);f.append('emo_mode',$('emo').value);f.append('emo_alpha',$('alpha').value);
  if($('spk').files[0])f.append('spk_audio',$('spk').files[0]);else f.append('voice',$('voice').value);
  const m=$('emo').value;
  if(m==='audio'&&$('emoAudio').files[0])f.append('emo_audio',$('emoAudio').files[0]);
  if(m==='vector')f.append('emo_vector',[...Array(8).keys()].map(i=>$('v'+i).value).join(','));
  if(m==='text'&&$('emoText').value.trim())f.append('emo_text',$('emoText').value.trim());
  $('go').disabled=true;setStatus('Đang tạo… (lượt đầu với giọng mới sẽ lâu hơn)');const t0=performance.now();
  try{
    const r=await fetch('/tts',{method:'POST',body:f,headers:hdr()});
    if(!r.ok){let msg=await r.text();try{msg=JSON.parse(msg).detail}catch(e){}throw new Error(msg)}
    const url=URL.createObjectURL(await r.blob());
    $('player').src=url;$('player').classList.remove('hide');$('player').play().catch(()=>{});
    $('dl').innerHTML=`<a href="${url}" download="tts.wav">Tải file WAV</a>`;
    const a=r.headers.get('X-Audio-Seconds'),s=r.headers.get('X-Inference-Seconds');
    setStatus(`Xong sau ${((performance.now()-t0)/1000).toFixed(1)} giây`+(a?` · audio dài ${a} giây · RTF ${r.headers.get('X-RTF')}`:''));
  }catch(e){setStatus('Lỗi: '+e.message,true)}finally{$('go').disabled=false}
};
loadVoices();
</script></body></html>"""


if __name__ == "__main__":
    print(f">> API docs: http://{cmd_args.host}:{cmd_args.port}/docs  ·  test page: http://{cmd_args.host}:{cmd_args.port}/")
    if not cmd_args.api_key and cmd_args.host not in ("127.0.0.1", "localhost"):
        print(">> WARNING: no --api_key set and listening on a public interface; anyone who can reach this port can use it.")
    uvicorn.run(app, host=cmd_args.host, port=cmd_args.port, workers=1)
