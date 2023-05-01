FROM amazon/aws-lambda-python

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG MODEL_DIR=./models
RUN mkdir $MODEL_DIR

ENV TRANSFORMERS_CACHE=$MODEL_DIR \
    TRANSFORMERS_VERBOSITY=error

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

RUN yum install git -y && yum -y install gcc-c++
COPY inf_requirements.txt inf_requirements.txt
RUN pip install -r inf_requirements.txt --no-cache-dir
RUN pip install --upgrade transformers
RUN pip install --upgrade tokenizer

RUN pip install --no-cache-dir awscli==1.22.54
RUN aws s3 cp s3://models-dvc2/trained_models/model.onnx ./models/model.onnx


COPY ./ ./
ENV PYTHONPATH "${PYTHONPATH}:./"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR
CMD [ "lambda_handler.lambda_handler"]


## below is the testing for running the application, excluding lambda
# FROM huggingface/transformers-pytorch-cpu:latest


# RUN pip install --no-cache-dir awscli==1.22.54
# COPY ./ /app
# WORKDIR /app

# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY


# # aws credentials configuration
# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
#     AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY


# # pulling the trained model
# #RUN dvc pull dvcfiles/trained_model.dvc
# RUN aws s3 cp s3://models-dvc2/trained_models/model.onnx ./models/model.onnx
# RUN pip install -r inf_requirements.txt
# RUN pip install --upgrade transformers
# RUN pip install --upgrade tokenizer

# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# # running the application
# #EXPOSE 8000
# #CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
