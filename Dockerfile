FROM python:3.12-slim
# ffmpeg for audio; pango for WeasyPrint (correct Indic text shaping in PDFs)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libpango-1.0-0 libpangoft2-1.0-0 \
    libharfbuzz-subset0 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
COPY eval ./eval
ENV ENV=prod PORT=8000
RUN useradd -m gh && mkdir -p /srv/app/var && chown -R gh /srv/app/var
USER gh
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
