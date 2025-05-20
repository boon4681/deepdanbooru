FROM python:3.11.4-slim

WORKDIR /main

COPY ./requirements.txt /main/requirements.txt

RUN python -m pip install -U pip setuptools wheel

RUN pip install --no-cache-dir --upgrade -r /main/requirements.txt

COPY ./app /main/app
COPY ./web_static /main/web_static

CMD ["fastapi", "run", "app/main.py", "--port", "4090"]