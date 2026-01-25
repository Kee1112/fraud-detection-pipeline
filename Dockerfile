FROM eclipse-temurin:17-jre

# Install Python 3.10 explicitly
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3-pip && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
