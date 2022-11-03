# FROM python:3.7.8-slim
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM pytorchlightning/pytorch_lightning:base-conda-py3.9-torch1.12

# remember to expose the port your app'll be exposed on.
EXPOSE 8080

RUN pip install -U pip

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt
RUN pip install streamlit

# copy into a directory of its own (so it isn't in the toplevel dir)
COPY . /app
WORKDIR /app

# run it!
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
