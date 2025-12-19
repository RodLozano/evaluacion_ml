FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /mlpontia

COPY . .

RUN pip install -r requirements.txt

CMD ["bash", "-lc", "cd Django && python manage.py makemigrations api && python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
