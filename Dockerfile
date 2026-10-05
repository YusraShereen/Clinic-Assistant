# NOT build-tested in the authoring environment (no Docker there) - see docs/NEXT_STEPS.md step F.
FROM python:3.11-slim
WORKDIR /srv
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/srv/.hf

# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Bake the embedding model into the image (needs internet at BUILD time, none at run time)
ARG EMBED_MODEL=intfloat/multilingual-e5-small
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer as S; S('${EMBED_MODEL}')"

COPY nlu ./nlu
COPY app ./app
COPY kb ./kb
COPY data/label_map.json ./data/label_map.json

# The fine-tuned NLU weights (~1.1 GB) are mounted at /models/nlu, not baked in.
ENV NLU_MODEL_DIR=/models/nlu LABEL_MAP=data/label_map.json KB_PATH=kb/faq.json \
    SYMPTOM_ROUTING=kb/symptom_routing.json EMBEDDER=e5 DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
