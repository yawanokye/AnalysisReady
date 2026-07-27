FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Fail the image build only if both the packaged asset and the embedded
# runtime fallback are unusable. Importing the module verifies that a valid
# component directory can be resolved before deployment.
RUN python -c "from statready.path_editor_component import component_asset_status; s=component_asset_status(); assert s['index_exists'], s; print(s)"

EXPOSE 10000

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]
