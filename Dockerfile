FROM python:3.10.7

RUN pip install --upgrade pip

ADD . /app

WORKDIR /app/

RUN pip install .

RUN pip install jupyter

RUN apt update && apt-get -y install libsndfile-dev

EXPOSE 8840

ENTRYPOINT ["jupyter", "notebook", "--ip=0.0.0.0" , "--port=8840", "--no-browser", "--allow-root"]

