FROM python:3.10-slim
WORKDIR /app

# ติดตั้ง libGL สำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libx11-dev \
    libxcb1 \
    libxcomposite1 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6

COPY requirements.txt .
COPY project.py .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "project.py"]