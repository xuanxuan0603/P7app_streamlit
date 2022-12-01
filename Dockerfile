FROM python:3.10.8-slim

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8501

CMD streamlit run app.py

