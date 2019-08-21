FROM python:3.6-slim

COPY app.py requirements.txt /app/
COPY data/export.pkl /app/data/

WORKDIR /app

RUN set -ex; \
	 apt-get update -y && apt-get install -y \
		python3-dev \
		gcc; \
    pip install --upgrade pip --no-cache-dir; \
	pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --no-cache-dir; \
	pip install -r requirements.txt --no-cache-dir; \
	apt-get autoremove -y \
		python3-dev \
		gcc; \
	apt-get clean; \
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

EXPOSE 8000

CMD ["python", "app.py"]
