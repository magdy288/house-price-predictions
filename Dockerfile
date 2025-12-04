FROM python:3.12-slim AS builder

# install requirements.txt, is exist in [src/api] path
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim-bookworm AS production

# Container path
WORKDIR /app

# copy our path -> [src/api/] to docker container path -> [/app]
COPY src/api/ .

# copy our all [model, preprocessor_data].pkl from our dir with [*] to select all files
# and copy it to [models/trained/] docker path for stand alone models folders in [/app]
COPY models/trained/*.pkl models/trained/

# the port
EXPOSE 8000

# command run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 