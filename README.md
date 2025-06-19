# Face Anonymization Pipeline

**Level Chosen: Level 1 - The Analyst**

This project implements a face detection and anonymization system using OpenCV with both DNN and Haar Cascade detectors. The system processes videos and outputs anonymized versions using different detection methods.

## Features

- **Multiple Detection Methods**: DNN, Haar Cascade, and Ensemble (combination of both)
- **Enhanced Preprocessing**: Histogram equalization and gamma correction for better detection
- **Multi-scale Detection**: Processes frames at different scales for improved accuracy
- **Non-Maximum Suppression**: Removes duplicate face detections
- **Automatic Output Generation**: Creates separate output files for each detection method

## Quick Start

### For Linux Users

1. **Build the Docker image:**
   ```bash
   docker build -t face-anonymizer .
   ```

2. **Run the container:**
   ```bash
   docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output_videos face-anonymizer
   ```

### For Windows Users

1. **Build the Docker image:**
   ```cmd
   docker build -t face-anonymizer .
   ```

2. **Run the container (Command Prompt):**
   ```cmd
   docker run -v %cd%/input:/app/input -v %cd%/output:/app/output_videos face-anonymizer
   ```

3. **Run the container (PowerShell):**
   ```powershell
   docker run -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output_videos face-anonymizer
   ```

## GPU Requirements

### Prerequisites
- **NVIDIA GPU** with CUDA support
- **NVIDIA Docker** installed on your system
- **CUDA-compatible drivers** (version 11.8 or higher)

### Setup NVIDIA Docker (One-time setup)

**Linux:**
```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Windows:**
- Install **Docker Desktop** with WSL2 backend
- Install **NVIDIA Container Toolkit** for Windows
- Ensure **CUDA drivers** are installed

### Verify GPU Access
Test GPU availability:
```bash
docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### 1. Prepare Your Environment

Create the required directories in your project folder:

**Linux/macOS:**
```bash
mkdir -p input output
```

**Windows:**
```cmd
mkdir input output
```

### 2. Add Your Video

Place your video file in the `input` directory and name it `video.mp4`:

**Linux/macOS:**
```bash
cp /path/to/your/video.mp4 input/video.mp4
```

**Windows:**
```cmd
copy "C:\path\to\your\video.mp4" input\video.mp4
```

### 3. Run the Processing

Use the docker run commands above. The system will automatically:
- Download required face detection models
- Process your video with multiple detection methods
- Save anonymized outputs to the `output` directory

## Output Files

The system generates two output files:
- `video_DNN_timestamp.mp4` - Processed with deep neural network detector (CPU optimized)
- `video_Haar_timestamp.mp4` - Processed with Haar cascade detector (CPU optimized)

## Advanced Usage

### Custom Video Filename

To process a video with a different name:

**Linux:**
```bash
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output_videos face-anonymizer python face_anonymizer.py /app/input/your_video.mp4
```

**Windows (Command Prompt):**
```cmd
docker run -v %cd%/input:/app/input -v %cd%/output:/app/output_videos face-anonymizer python face_anonymizer.py /app/input/your_video.mp4
```

**Windows (PowerShell):**
```powershell
docker run -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output_videos face-anonymizer python face_anonymizer.py /app/input/your_video.mp4
```

### Interactive Mode

To run the container interactively for debugging:

**Linux:**
```bash
docker run -it -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output_videos face-anonymizer bash
```

**Windows (Command Prompt):**
```cmd
docker run -it -v %cd%/input:/app/input -v %cd%/output:/app/output_videos face-anonymizer bash
```

**Windows (PowerShell):**
```powershell
docker run -it -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output_videos face-anonymizer bash
```

## Troubleshooting

### Common Issues

1. **"Video file not found" error:**
   - Ensure your video is in the `input` directory
   - Check that the filename is `video.mp4` or specify the correct path

2. **Permission errors (Linux/macOS):**
   ```bash
   sudo chown -R $USER:$USER output/
   ```

3. **Docker build fails:**
   - Ensure Docker is running
   - Check internet connection for downloading dependencies
   - Try building with `--no-cache` flag:
     ```bash
     docker build --no-cache -t face-anonymizer .
     ```

4. **Output directory empty:**
   - Check Docker logs: `docker logs <container_id>`
   - Verify input video format is supported (MP4, AVI, MOV)

### System Requirements

- **Docker**: Version 20.10 or higher
- **RAM**: Minimum 2GB available for the container
- **Storage**: Sufficient space for input video + 2x output videos
- **CPU**: Multi-core recommended for faster processing

### Supported Video Formats

- MP4 (recommended)
- AVI
- MOV
- WMV
- FLV

## Technical Details

### Detection Methods

1. **DNN (Deep Neural Network)**:
   - Uses OpenCV's pre-trained SSD MobileNet model
   - More accurate in varied lighting conditions
   - Better at detecting faces at different angles
   - CPU optimized for compatibility

2. **Haar Cascade**:
   - Traditional computer vision approach
   - Faster processing (CPU optimized)
   - Single cascade for frontal faces only
   - Optimized parameters for speed

### Performance

Processing time depends on:
- Video resolution and length
- Number of faces in the video
- Available CPU cores
- Detection method used

Typical processing speeds:
- 720p video: 2-5x real-time
- 1080p video: 1-3x real-time
- 4K video: 0.5-1x real-time