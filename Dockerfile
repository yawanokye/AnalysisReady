FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OPENAI_VISION_MODEL=gpt-5.4-nano \
    OPENAI_VISION_FALLBACK_MODEL=gpt-5.6-luna \
    OPENAI_VISION_DETAIL=low \
    OPENAI_VISION_FALLBACK_DETAIL=high \
    OPENAI_VISION_MAX_DIMENSION=1800 \
    OPENAI_VISION_JPEG_QUALITY=88 \
    DEEPSEEK_BASE_URL=https://api.deepseek.com \
    AGENT_REASONING_MODEL=deepseek-v4-flash \
    AGENT_REASONING_FALLBACK_MODEL=deepseek-v4-pro \
    AGENT_MAX_OUTPUT_TOKENS=12000

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Catch missing Python packages during the image build rather than at web-service startup.
RUN python -m statready.dependency_check

# Fail the image build only if both the packaged asset and the embedded
# runtime fallback are unusable. Importing the module verifies that a valid
# component directory can be resolved before deployment.
RUN python -c "from statready.path_editor_component import component_asset_status; s=component_asset_status(); assert s['index_exists'], s; print(s)"

EXPOSE 10000

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]
