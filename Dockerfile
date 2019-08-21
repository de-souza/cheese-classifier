FROM python:3.6-slim

RUN apt-get update
RUN apt-get install -y python3-dev gcc

COPY requirements.txt requirements.txt

RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip install -r requirements.txt

ADD app.py app.py
ADD data/export.pkl data/export.pkl

EXPOSE 8000

CMD ["python", "app.py"]
