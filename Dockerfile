FROM python:3.9-slim

# avoid .pyc and __pycache__ files and directories
ENV PYTHONDONTWRITEBYTECODE=1
# do not buffer stdout/stderr
ENV PYTHONUNBUFFERED=1

# --- Create non-root user ---
ARG APP_USER=appuser
ARG APP_UID=10001
ARG APP_GID=10001

RUN groupadd -g "${APP_GID}" "${APP_USER}"; \
    useradd  -m -u "${APP_UID}" -g "${APP_GID}" -s /bin/bash "${APP_USER}"

# --- Workdir ---
WORKDIR /app

# --- Install dependencies (best layer caching practice) ---
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip; \
    pip install --no-cache-dir -r /app/requirements.txt 

RUN pip list

# --- Copy app code + entrypoint ---
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY src /app

# Ensure entrypoint is executable and app files owned by non-root user
RUN chmod +x /usr/local/bin/docker-entrypoint.sh; \
    chown -R "${APP_UID}:${APP_GID}" /app

# Drop privileges
USER ${APP_UID}:${APP_GID}

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]