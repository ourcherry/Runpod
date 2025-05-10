FROM python:3.10-slim

# 설치
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 엔트리포인트 지정
CMD ["python", "handler.py"]
