FROM python:3.12
WORKDIR /rag

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/

EXPOSE 8501

COPY poetry.lock pyproject.toml ./

RUN poetry install  --no-root && rm -rf $POETRY_CACHE_DIR

COPY rag/__init__.py TextSplitter.py GenerateEmb.py .env /rag/

RUN poetry install

ENTRYPOINT ["poetry", "run", "streamlit", "run", "__init__.py", "--server.port=8501", "--server.address=0.0.0.0"]
