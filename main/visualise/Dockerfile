FROM python:3.9-slim-buster

COPY requirements.txt /app/
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir pip-tools \
    && pip-sync requirements.txt --pip-args "--no-cache-dir"

COPY .streamlit ./.streamlit/
COPY data ./data/
COPY images ./images/
COPY scripts ./scripts/
COPY app.py /app/

EXPOSE 8080 2222
CMD streamlit run app.py
