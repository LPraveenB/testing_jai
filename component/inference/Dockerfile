ARG PYTHON_VERSION=""
FROM python:${PYTHON_VERSION}

# Install jq
RUN apt-get update &&     apt-get install -y jq &&     apt-get clean

# Install pip.
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py &&     python get-pip.py &&     pip3 install setuptools &&     rm get-pip.py
RUN python3 -m pip install --upgrade pip

COPY build.json /app/
# COPY ../requirements.txt /app/

#COPY ${WORKSPACE}/component/requirements.txt /app/
#COPY ${WORKSPACE}/component/build.json /app/
# Install packages using versions from the config file
WORKDIR /app

RUN python3 -m pip install -r requirements.txt

# Remove the JSON config file
RUN rm -rf build.json

# Remove additional files or directories if needed
RUN rm -rf inference

RUN mkdir -p inference
COPY * inference/
WORKDIR inference
RUN ls

ENTRYPOINT ["bash", "build_inference.sh"]