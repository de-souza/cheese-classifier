FROM python:3.6-slim

RUN apt-get update
RUN apt-get install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install fastai

# Install starlette and uvicorn
RUN pip install starlette uvicorn python-multipart aiohttp

ADD app.py app.py
ADD data/export.pkl data/export.pkl

EXPOSE 8000

# Start the server
CMD ["python", "app.py"]
