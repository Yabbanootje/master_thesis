FROM python:3.10

ADD test.py .

RUN pip install omnisafe

CMD ["python", "./test.py"]