FROM eclipse-temurin:11-jre-jammy

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip + setuptools + wheel (important for numpy)
RUN pip3 install --upgrade pip setuptools wheel

# Install numpy FIRST (binary wheel)
RUN pip3 install --no-cache-dir numpy==1.26.4

# Install Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefer-binary -r requirements.txt

# Copy app + models
COPY src /app/src
COPY models /app/models_a

EXPOSE 8000

CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
