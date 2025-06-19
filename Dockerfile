FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    numpy==1.24.3

WORKDIR /app

COPY face_anonymizer.py .

RUN mkdir -p input output_videos

VOLUME ["/app/input", "/app/output_videos"]

CMD ["python", "face_anonymizer.py", "/app/input/video.mp4"]