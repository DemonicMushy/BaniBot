FROM python:3.7-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
         libgl1-mesa-glx \
         build-essential \
         software-properties-common \
         nano \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

RUN yes | pip install \
    wheel \
    uwsgi \
    flask==1.1.2 \
    Bani==0.6.1 && \
    pip cache purge

RUN python -m spacy download en_core_web_md

RUN mkdir latest_model faqStore

COPY faqStore_1612 ./faqStore/
COPY latest_model3 ./latest_model/
COPY flaskApp.py flaskapp.ini ./


CMD ["uwsgi", "flaskapp.ini"]
