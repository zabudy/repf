FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY requirements-local.txt .
RUN pip install --no-cache-dir -r requirements-local.txt "celery[redis]>=5.4,<6" "asyncpg>=0.29,<1"
RUN useradd --create-home --uid 10001 repf
COPY --chown=repf:repf backend backend
COPY --chown=repf:repf scripts scripts
COPY --chown=repf:repf frontend frontend
COPY --chown=repf:repf data/import data/import
RUN mkdir -p data/processed logs && chown -R repf:repf data/processed logs
USER repf
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
