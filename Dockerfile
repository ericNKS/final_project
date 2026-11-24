FROM docker.io/library/python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    TRAINING_BACKEND=gpu \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libnuma1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

ARG APP_UID=1000
ARG APP_GID=1000
RUN groupadd --gid "$APP_GID" app \
    && useradd --uid "$APP_UID" --gid "$APP_GID" --create-home app \
    && mkdir -p /app/resultados /app/data /home/app/.cache \
    && chown -R app:app /app /home/app

COPY --chown=app:app main.py alembic.ini pytest.ini ./
COPY --chown=app:app src ./src
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app tests ./tests
COPY --chown=app:app docs ./docs

USER app
ENV HOME=/home/app
EXPOSE 8000

ENTRYPOINT ["python", "main.py"]
CMD ["serve", "--host", "::", "--port", "8000"]
