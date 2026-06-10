# Lift Wing OpenAPI Server

Lightweight Apache httpd service that serves the Lift Wing OpenAPI specification
files as static content. Unlike the other model servers in this repo, this is
**not** a KServe model server — it does not use PyTorch, model volumes, or the
KServe framework.

## What it does

- Serves the aggregated [`docs/openapi.yaml`](../../../docs/openapi.yaml) and all
  per-model OpenAPI spec files at `/v1/openapi.yaml`
- Adds CORS headers so browser-based tools
  (Swagger UI, [MediaWiki RestSandbox](https://www.mediawiki.org/wiki/API/REST_Sandbox))
  can fetch the specs cross-origin
- The `openapi.yaml` file uses `$ref` references to sibling files
  (e.g. `./langid.yaml`), so the entire `docs/` directory is served

## Architecture

Based on the
[statictendril/miscweb](https://gitlab.wikimedia.org/repos/sre/miscweb/statictendril)
pattern. No custom Python code — just Apache httpd with config files.

| File | Purpose |
|---|---|
| [`httpd.conf`](httpd.conf) | Main Apache config — loads modules (`mod_headers`, `mod_mime`, `mod_rewrite`, etc.) and includes the vhost |
| [`liftwingspec.conf`](liftwingspec.conf) | VirtualHost on `*:8080` — `DocumentRoot /srv/app/docs`, CORS headers, `ServerAlias api.wikimedia.org` |
| [`entrypoint.sh`](entrypoint.sh) | Starts Apache with `apache2 -d /srv/app -f /srv/app/apache2.conf -DFOREGROUND` |
| [`blubber.yaml`](../../../.pipeline/liftwing_openapi_server/blubber.yaml) | Blubber build config — `docker-registry.wikimedia.org/httpd` base, copies `docs/` and config files into `/srv/app/` |

### How the config fits together in blubber

```
/srv/app/
├── apache2.conf          ← copied from httpd.conf
│   ├── Loads mod_headers, mod_mime, mod_rewrite, ...
│   └── IncludeOptional liftwingspec.conf
├── liftwingspec.conf     ← VirtualHost with DocumentRoot, CORS, ServerAlias
├── docs/                 ← the OpenAPI YAML files (openapi.yaml + per-model specs)
│   ├── openapi.yaml
│   ├── langid.yaml
│   ├── edit-check.yaml
│   └── ...
├── entrypoint.sh
```

## Running locally

Build and start the server:

```bash
docker compose build liftwing-openapi-server
docker compose up -d liftwing-openapi-server
```

Test it:

```bash
# Main aggregated spec
curl http://localhost:8081/openapi.yaml
```

### Local MediaWiki integration

With MediaWiki running at `http://127.0.0.1:8080`, add this to
`LocalSettings.php` to test the REST Sandbox:

```php
$wgRestSandboxSpecs['lw-openapi'] = [
    'url' => 'http://localhost:8081/openapi.yaml',
    'name' => 'Lift Wing',
    'group' => 'Lift Wing',
];
```

Then visit `http://127.0.0.1:8080/w/index.php/Special:RestSandbox`.

## Deployment

The image is built via Jenkins and published as
`inference-services-liftwing-openapi-server:stable`. Since this is not a
KServe InferenceService, it is deployed as a plain Kubernetes Deployment
using the [liftwing-openapi-server chart](https://gerrit.wikimedia.org/r/plugins/gitiles/operations/deployment-charts/+/refs/heads/master/helmfile.d/ml-services/liftwing-openapi-server/values.yaml) and following the standard procedure of deployments based on the wikitech page: [Machine Learning/LiftWing/Deploy](https://wikitech.wikimedia.org/wiki/Machine_Learning/LiftWing/Deploy#How_to_deploy) .

This will be deployed and exposed into this public url: `https://api.wikimedia.org/service/lw/specs/openapi.yaml` which is configured in the RestSandbox at the `InitialiseSettings.php` in the mediawiki-config repo [wgRestSandboxSpecs](https://gerrit.wikimedia.org/r/plugins/gitiles/operations/mediawiki-config/+/refs/heads/master/wmf-config/InitialiseSettings.php#13154).
