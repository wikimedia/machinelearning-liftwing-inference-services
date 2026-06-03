# Lift Wing OpenAPI Server

Serves the Lift Wing OpenAPI specs via Apache httpd, following the miscweb/statictendril pattern.

## Local

```bash
docker compose build liftwing-openapi-server && docker compose up liftwing-openapi-server
curl http://localhost:8081/openapi.yaml
```

## How it works

Apache httpd with a custom config serving `docs/` as static files with CORS headers. Built from the WMF httpd base image, matching the statictendril miscweb pattern.
