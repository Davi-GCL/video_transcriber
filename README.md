Rodar
# CPU (mais compatível, mais lento)
set USE_CUDA=0
python transcrever.py ".\meu_video.mp4" pt

# GPU (se tiver CUDA funcional)
set USE_CUDA=1
python transcrever.py ".\meu_video.mp4" pt

# Detectar idioma automaticamente
python transcrever.py ".\meu_video.mp4" auto
