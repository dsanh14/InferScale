FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs results

EXPOSE 8000 8001 8002 8003 8004 8005

CMD ["uvicorn", "app.router.main:app", "--host", "0.0.0.0", "--port", "8000"]
