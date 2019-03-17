FROM python:3.6-slim

RUN apt-get update
RUN apt-get install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD cheese-classifier.py cheese-classifier.py
ADD export.pkl export.pkl

EXPOSE 8008

# Start the server
CMD ["python", "cheese-classifier.py"]
