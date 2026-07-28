# Render environment setup

The application already contains all non-secret defaults in both `render.yaml` and the Docker image. Add only the following secrets in Render:

- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`

## Render dashboard

1. Open the `statready-ai` service.
2. Select **Environment**.
3. Add `OPENAI_API_KEY` and paste the OpenAI API key.
4. Add `DEEPSEEK_API_KEY` and paste the DeepSeek API key.
5. Save changes.
6. Select **Manual Deploy**, then **Clear build cache & deploy**.

Do not commit either key to GitHub, `.env`, `render.yaml`, or the Dockerfile.

## Non-secret defaults

```text
OPENAI_VISION_MODEL=gpt-5.4-nano
OPENAI_VISION_FALLBACK_MODEL=gpt-5.6-luna
OPENAI_VISION_DETAIL=low
OPENAI_VISION_FALLBACK_DETAIL=high
OPENAI_VISION_MAX_DIMENSION=1800
OPENAI_VISION_JPEG_QUALITY=88
DEEPSEEK_BASE_URL=https://api.deepseek.com
AGENT_REASONING_MODEL=deepseek-v4-flash
AGENT_REASONING_FALLBACK_MODEL=deepseek-v4-pro
AGENT_MAX_OUTPUT_TOKENS=12000
```

These defaults work whether the service is created through the Render Blueprint or directly from the Dockerfile.
