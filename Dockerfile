# Użycie oficjalnego obrazu Pythona
FROM python:3.9-slim

# Ustawienie katalogu roboczego w kontenerze
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]
