FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend /app/backend
COPY public /app/public

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["/usr/local/bin/python", "backend/main.py"]
