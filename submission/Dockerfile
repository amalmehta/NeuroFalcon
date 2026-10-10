# EvalAI submission for FALCON M2. Build from the project root:
#   docker build -f submission/Dockerfile -t neurofalcon-m2 .
# Mandatory lines follow the harness's decoder_demos/template.Dockerfile.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/snel-repo/falcon-challenge.git
RUN pip install -e falcon-challenge

ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH "/tmp/ground_truth.pkl"

ENV EVALUATION_LOC remote

ADD ./neurofalcon/__init__.py neurofalcon/__init__.py
ADD ./neurofalcon/sessions.py neurofalcon/sessions.py
ADD ./neurofalcon/gru.py neurofalcon/gru.py
ADD ./submission/decoder.pt data/decoder.pt
ADD ./submission/decode.py decode.py

ENV SPLIT "m2"
ENV PHASE "test"
ENV BATCH_SIZE 1
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path data/decoder.pt --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]
