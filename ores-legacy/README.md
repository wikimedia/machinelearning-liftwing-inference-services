# Running locally
_Running the service locally will just run the FastAPI app but it will not have access to LiftWing as the application is intended to be run through our Kubernetes environment_

Build the production image:
```bash
docker build --target production -f .pipeline/ores-legacy/blubber.yaml -t ores:prod .
```

Run the container binding the port 8000 to the container port 80
```bash
docker run -d --name mycontainer -p 8000:80 ores:prod
```

The FastAPI app is now running on http://localhost:8000,
You can check the Swagger UI at http://localhost:8000/docs or redoc at http://localhost:8000/redoc

In order to test the application end to end with LiftWing, one can run the application in one of the statboxes by doing
the following:

- ssh into a statbox
- `git clone "https://gerrit.wikimedia.org/r/machinelearning/liftwing/inference-services"`
- `export https_proxy=http://webproxy:8080`
- `make test-server` will create a virtual environment, set up dependencies and
   run Uvicorn with basic settings.
- Access the application and make calls from your local machine at http://localhost:8000
