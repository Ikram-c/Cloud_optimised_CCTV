FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml LICENSE README.md ./
COPY src ./src
COPY config.docker.yaml ./config.yaml

RUN pip install --no-cache-dir ".[ui]"

VOLUME ["/data"]
EXPOSE 8432

CMD ["cctv-zarr-ui", "--config", "config.yaml"]
