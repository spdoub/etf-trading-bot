# ETF bot: background trading scheduler + Streamlit dashboard on $PORT.
# Mount a Railway volume at /data and set DB_PATH=/data/etf_bot.db.
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        tini \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
ENV DB_PATH=/data/etf_bot.db

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/railway_start.sh

EXPOSE 8501

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["./scripts/railway_start.sh"]
