# ---- 1. build the React app ----
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---- 2. Python API + the built frontend, one container ----
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ml ./ml
COPY backend ./backend
COPY models ./models
COPY --from=web /web/dist ./frontend/dist
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]