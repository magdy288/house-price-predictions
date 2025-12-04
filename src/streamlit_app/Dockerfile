FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim-bookworm AS production

WORKDIR /app

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]