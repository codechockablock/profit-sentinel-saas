FROM python:3.12-slim

WORKDIR /app

COPY app/main.py .

RUN pip install fastapi uvicorn python-multipart pandas openpyxl boto3
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]