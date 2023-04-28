FROM python:3.8-buster
COPY ./ /app
WORKDIR /app
RUN pip install -r inf_requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]