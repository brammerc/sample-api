#!/bin/bash

# 5 sec delay to give docker-stack-wait time to see the stack before an immediate failure 
sleep 5

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
WORKERS="${WORKERS:-2}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# This adds a random variation to each worker's restart point (staggered between
# MAX_REQUESTS and MAX_REQUESTS + JITTER). This ensures workers don't all reboot 
# simultaneously, which would drop traffic while models reload.
MAX_REQUESTS=$(( WORKERS * 100 ))
MAX_REQUESTS_JITTER=$(( WORKERS * 10 ))

export MODEL_ACTIVE_VERSION="${MODEL_ACTIVE_VERSION:v1}"
export SHADOW_MODELS="${SHADOW_MODELS:-}"

# Rule of thumb: gunicorn workers <= CPU cores * 2
{
gunicorn --worker-class uvicorn.workers.UvicornWorker \
         --bind "${HOST}:${PORT}" \
         --workers "${WORKERS}" \
         --user 10001 \
         --group 10001 \
         --max-requests "${MAX_REQUESTS}" \
         --max-requests-jitter "${MAX_REQUESTS_JITTER}" \
         --log-level "${LOG_LEVEL}" \
         --access-logfile - \
         --error-logfile - \
         app:app 
} 2>&1

exit $?
