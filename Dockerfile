FROM python:3.10
ADD . /app
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --no-cache-dir --user -r requirements.txt
RUN pip3 install --ignore-installed uvicorn[standard]>=0.18.3
RUN pip3 install python-multipart
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'
ENTRYPOINT ["uvicorn", "main:app","--reload", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]