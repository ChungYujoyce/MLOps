FROM huggingface/transformers-pytorch-cpu:latest
COPY ./ /app
WORKDIR /app

RUN pip install "dvc[gdrive]"
RUN pip install -r inf_requirements.txt

# initialise dvc
RUN dvc init --no-scm
# configuring remote server in dvc
RUN dvc remote add -d myremote gdrive://1WwaXe32DzGoTm5stVzlG_SERy43SW1rz
RUN dvc remote modify myremote gdrive_use_service_account true
RUN dvc remote modify myremote gdrive_service_account_json_file_path creds.json
RUN dvc pull dvcfiles/trained_model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]