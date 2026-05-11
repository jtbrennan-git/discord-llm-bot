FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache buster: force rebuild
ARG CACHEBUST=1
RUN echo "Cache bust: $CACHEBUST"

COPY . .

CMD ["python", "-m", "bot.main"]
