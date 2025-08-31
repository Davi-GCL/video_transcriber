from faster_whisper import WhisperModel
import os, sys

if len(sys.argv) < 2:
    print("Uso: python transcrever.py <arquivo_de_video_ou_audio> [pt|en|auto]")
    sys.exit(1)

arquivo = sys.argv[1]
idioma = sys.argv[2] if len(sys.argv) > 2 else "pt"  # "pt", "en" ou "auto"

# Modelos: tiny, base, small, medium, large-v3 (quanto maior, mais preciso e mais lento)
modelo = "large-v3"  # bom balanço pt-BR. Troque para "large-v3" se quiser a máxima precisão.

# device="cuda" usa GPU NVIDIA; "cpu" para CPU
model = WhisperModel(modelo, device="cuda" if os.getenv("USE_CUDA","1")=="1" else "cpu", compute_type="float16" if os.getenv("USE_CUDA","1")=="1" else "int8")

segments, info = model.transcribe(
    arquivo,
    language=None if idioma=="auto" else idioma,
    vad_filter=True,                # remove silêncio/ruído
    vad_parameters=dict(min_silence_duration_ms=500),
    beam_size=5,
)

# Salva SRT e TXT
base = os.path.splitext(arquivo)[0]
with open(base + ".txt","w",encoding="utf-8") as f_txt, open(base + ".srt","w",encoding="utf-8") as f_srt:
    i = 1
    for seg in segments:
        # TXT contínuo
        f_txt.write(seg.text.strip()+" ")
        # SRT com timestamps
        def ts(t):
            h = int(t//3600); m=int((t%3600)//60); s=t%60
            return f"{h:02}:{m:02}:{s:06.3f}".replace(".",",")
        f_srt.write(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{seg.text.strip()}\n\n")
        i += 1

print("Concluído:")
print(" -", base + ".txt")
print(" -", base + ".srt")
