# Docker Configuration

This directory contains Docker configuration files for containerized execution of the project.

## Files

### `Dockerfile`
**Multi-stage Docker image definition for the project.** This file:
- Defines the complete build process for creating a Docker image
- Installs all system dependencies and Python packages
- Sets up the execution environment
- Enables reproducible experiments across different machines

**Image Details:**
- **Base Image**: `python:3.12-slim`
  - Minimal Debian-based Python installation
  - Smaller image size than full Python image
  - Includes essential build tools
  
- **Working Directory**: `/app`
  - All project files mounted/copied here
  - PYTHONPATH set to `/app/src`

**Build Stages:**

**Stage 1: System Dependencies**
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \    # C/C++ compilers for native extensions
    ffmpeg \            # Video processing (for visualization)
    libsm6 \            # System libraries for OpenCV
    libxext6 \          # X11 extensions
    dos2unix \          # Line ending conversion
    jq \                # JSON processing in shell scripts
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
```

**Why these packages?**
- `build-essential`: Compile Python packages with native code (NumPy, PyTorch)
- `ffmpeg`: Create GIF animations from video frames
- `libsm6, libxext6`: Required by OpenCV and matplotlib
- `dos2unix`: Handle files from Windows (line endings)
- `jq`: Parse JSON in shell scripts

**Stage 2: Python Dependencies**
```dockerfile
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
```

**Optimization:** Separate layer for requirements
- If requirements.txt unchanged → cached layer reused
- Only project code changes → fast rebuild
- Saves time during development

**Stage 3: Application Code**
```dockerfile
COPY . /app
ENV PYTHONPATH=/app/src
```

**Sets up:**
- Project files copied into container
- Python import path configured
- Ready to run experiments

**Stage 4: User Configuration (Optional)**
```dockerfile
# RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
# USER appuser
```

**Security best practice:**
- Don't run as root inside container
- Create non-privileged user
- Currently commented out for compatibility

## Building the Docker Image

### Basic Build
```bash
docker build -f ./docker/Dockerfile -t student_mechanism_design .
```

**Options:**
- `-f ./docker/Dockerfile`: Specify Dockerfile location
- `-t student_mechanism_design`: Tag (name) the image
- `.`: Build context (project root directory)

**Build Time:** ~5 minutes first time, ~10 seconds for rebuilds (with cache)

**Disk Space:** ~2GB for complete image

### Build with No Cache
```bash
docker build --no-cache -f ./docker/Dockerfile -t student_mechanism_design .
```

**Use when:**
- Want completely fresh build
- Suspect cache corruption
- Testing reproducibility

### Multi-Platform Build
```bash
docker buildx build --platform linux/amd64,linux/arm64 -f ./docker/Dockerfile -t student_mechanism_design .
```

**Use for:**
- Cross-platform compatibility
- ARM Macs + x86 Linux servers
- Distribution to different architectures

## Running Containers

### Interactive Shell
```bash
docker run --rm -it \
    --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design
```

**Explanation:**
- `--rm`: Remove container after exit (cleanup)
- `-it`: Interactive terminal (can type commands)
- `--gpus=all`: Access all GPUs (requires nvidia-docker)
- `--mount`: Bind mount project directory (changes reflected on host)
- `student_mechanism_design`: Image name

**Inside container:**
```bash
# You're now in /app directory
ls  # See project files
python src/main.py --config ...
./scripts/run_experiment.sh smoke_train
exit  # When done
```

### Run Single Command
```bash
docker run --rm --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    python src/main.py --config src/configs/experiments/smoke_train/config.yml
```

**Difference from interactive:**
- No `-it` flag
- Command specified after image name
- Container exits after command completes

### Run Script
```bash
docker run --rm --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    bash -c "./scripts/run_experiment.sh smoke_train"
```

**Using bash -c:**
- Execute shell script inside container
- Can use shell features (pipes, redirects, etc.)
- Good for complex command sequences

### Run Tests
```bash
docker run --rm \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    pytest test/ -v
```

**No GPU needed for tests:**
- Tests don't require GPU
- Faster startup without GPU initialization
- Can run on CPU-only machines

## GPU Support

### Requirements
- **NVIDIA GPU** on host machine
- **NVIDIA drivers** installed on host
- **nvidia-docker** or **Docker with GPU support**

### Check GPU Access
```bash
# Inside container
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   45C    P8    10W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### Without GPU
```bash
# Run on CPU only (slower but works)
docker run --rm \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    python src/main.py --config src/configs/experiments/smoke_train/config.yml
```

**Note:** PyTorch automatically uses CPU if GPU unavailable

## Volume Mounts

### Bind Mount (Development)
```bash
--mount type=bind,src=$PWD,dst=/app
```

**Pros:**
- Changes on host reflected in container immediately
- Good for development (edit files, run in container)
- No data duplication

**Cons:**
- Performance can be slower (especially on Mac/Windows)
- File permission issues possible

### Volume Mount (Production)
```bash
# Create named volume
docker volume create experiment_data

# Use volume
docker run --rm -v experiment_data:/app/data student_mechanism_design
```

**Pros:**
- Better performance
- Docker manages volume lifecycle
- Shareable between containers

**Cons:**
- Data not directly accessible on host
- Need docker cp to extract files

### Read-Only Mount
```bash
--mount type=bind,src=$PWD/src,dst=/app/src,readonly
```

**Use case:** Prevent accidental modification of source code

## Environment Variables

### Set Environment Variables
```bash
docker run --rm \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e OMP_NUM_THREADS=4 \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design \
    python src/main.py --config ...
```

**Common Variables:**
- `CUDA_VISIBLE_DEVICES`: Control which GPUs to use
- `OMP_NUM_THREADS`: Limit CPU threads
- `PYTHONPATH`: Python import path (already set in Dockerfile)
- `WANDB_API_KEY`: WandB authentication

### From File
```bash
docker run --rm \
    --env-file .env \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design
```

**.env file:**
```
CUDA_VISIBLE_DEVICES=0,1
OMP_NUM_THREADS=8
WANDB_API_KEY=your_key_here
```

## Image Management

### List Images
```bash
docker images student_mechanism_design
```

### Remove Image
```bash
docker rmi student_mechanism_design
```

### Prune Unused Images
```bash
docker image prune -a
```

**Warning:** Removes all unused images, not just this project

### Save Image to File
```bash
docker save student_mechanism_design > student_mechanism_design.tar
```

**Use case:** Transfer image to another machine without Docker Hub

### Load Image from File
```bash
docker load < student_mechanism_design.tar
```

## Container Management

### List Running Containers
```bash
docker ps
```

### List All Containers (including stopped)
```bash
docker ps -a
```

### Stop Container
```bash
docker stop <container_id>
```

### Remove Container
```bash
docker rm <container_id>
```

### Clean Up All Stopped Containers
```bash
docker container prune
```

## Troubleshooting

### Problem: Build Fails with "requirements.txt not found"
**Solution:** Make sure you're running docker build from project root:
```bash
pwd  # Should be .../student-mechanism-design
docker build -f ./docker/Dockerfile -t student_mechanism_design .
```

### Problem: GPU not accessible in container
**Check:**
```bash
# On host
nvidia-smi  # Should work

# Docker with GPU support installed?
docker run --rm --gpus=all nvidia/cuda:11.0-base nvidia-smi
```

**Solution:** Install nvidia-docker:
```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Problem: Permission denied when accessing mounted files
**Solution:** Run with user ID mapping:
```bash
docker run --rm -it \
    --user $(id -u):$(id -g) \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design
```

### Problem: Container runs out of memory
**Solution:** Increase Docker memory limit:
```bash
# Docker Desktop: Settings → Resources → Memory
# Or use --memory flag:
docker run --rm --memory=8g --mount type=bind,src=$PWD,dst=/app student_mechanism_design
```

### Problem: CUDA out of memory in container
**Solution:** Limit GPU memory or batch size:
```bash
docker run --rm \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=128 \
    --gpus=all \
    --mount type=bind,src=$PWD,dst=/app \
    student_mechanism_design
```

Or modify config to use smaller batch size.

## Best Practices

**1. Use .dockerignore**
Create `.dockerignore` in project root:
```
__pycache__/
*.pyc
.git/
.vscode/
*.log
logs/
*.pt
```

**Why:** Faster builds, smaller images

**2. Layer Caching**
- Put rarely-changing steps first (system deps)
- Put frequently-changing steps last (application code)
- Leverage Docker's layer cache for faster rebuilds

**3. Multi-Stage Builds**
For production, use multi-stage builds:
```dockerfile
# Build stage
FROM python:3.12-slim as builder
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim
COPY --from=builder /root/.local /root/.local
COPY . /app
```

**4. Security**
- Don't run as root (uncomment USER directive)
- Don't include secrets in image
- Use specific base image versions (not 'latest')
- Regularly update base images

## Tips for Students

1. **Build once, run many times**: After initial build, running is fast
2. **Use bind mounts for development**: Edit code on host, run in container
3. **Test in Docker before submission**: Ensures reproducibility
4. **Save experiment logs**: Bind mount logs directory to persist data
5. **GPU optional for testing**: CPU mode works for smoke tests
6. **Clean up regularly**: `docker system prune` to free disk space
7. **Learn docker basics**: Official Docker tutorial is excellent
8. **Document your setup**: Note any special Docker flags needed

## CI/CD Integration

The project includes GitHub Actions workflow for Docker:

**`.github/workflows/docker.yml`** (if exists)
- Builds Docker image on every push
- Runs tests in container
- Validates Dockerfile syntax
- Ensures reproducibility

**Local testing before push:**
```bash
# Build
docker build -f ./docker/Dockerfile -t student_mechanism_design .

# Test
docker run --rm student_mechanism_design pytest test/ -v

# If both succeed, push is likely to pass CI
```
