FROM python:3.10

WORKDIR /app

COPY . /app

RUN chmod +x /app/start.sh

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash", "/app/start.sh"]
