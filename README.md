

```bash
docker run -d \
  -v "/${PWD}/model:/app/model" \
  -v "/${PWD}/logs:/app/logs" \
  -p 8080:8080 \
  -e HOST=0.0.0.0 \
  -e PORT=8080 \
  -e WORKERS=3 \
  -e LOG_LEVEL=debug \
  -e MODEL_ACTIVE_VERSION=v1 \
  --name sample-api-container \
  sample-api

docker logs -f sample-api-container

python client.py ./mle-project-challenge-2/data/future_unseen_examples.csv


docker run -d \
  -v "/${PWD}/model:/app/model" \
  -v "/${PWD}/logs:/app/logs" \
  -p 8080:8080 \
  -e HOST=0.0.0.0 \
  -e PORT=8080 \
  -e WORKERS=3 \
  -e LOG_LEVEL=debug \
  -e MODEL_ACTIVE_VERSION=v1 \
  -e SHADOW_MODELS=v2 \
  --name sample-api-container \
  sample-api
```