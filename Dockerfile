FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

LABEL org.grand-challenge.api-method="invoke"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

WORKDIR /opt/algorithm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/algorithm/requirements.txt
RUN pip install --no-cache-dir -r /opt/algorithm/requirements.txt

COPY app.py /opt/algorithm/app.py
COPY src /opt/algorithm/src

RUN groupadd -r algorithm && \
    useradd -r -g algorithm -d /opt/algorithm -s /sbin/nologin algorithm && \
    chown -R algorithm:algorithm /opt/algorithm

USER algorithm

EXPOSE 8080

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
