FROM python:3.10-slim

# Install Java (required for Spark)
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Set Java env
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set working dir
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + models
COPY src ./src
COPY models ./models

# Expose port
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
