FROM python:3.12-slim
WORKDIR /app
COPY .  /app
RUN apt-get update && apt-get install build-essential -y
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install dos2unix -y
RUN apt install -y jq
RUN pip install -r requirements.txt
RUN gcc --version
RUN dos2unix /app/run_main.sh
RUN chmod a+x /app/run_main.sh
ENV PYTHONPATH=/app/src
ENTRYPOINT ["/app/run_main.sh"]