# Running locally
_Running the service locally will just run the FastAPI app but it will not have access to LiftWing as the application is intended to be run through our Kubernetes environment_

Build the production image:
```bash
docker build --target production -f .pipeline/ores-migration/blubber.yaml -t ores:prod .
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
- upload/scp files under ores-migration directory
- create a virtual environment and install the requirements.txt file
- run the application using uvicorn and setting the environment variable for the LIFTWING_URL
  > LIFTWING_URL=https://inference.svc.codfw.wmnet:30443 uvicorn app.main:app --reload --port 8000
- use ssh tunneling to access the application from your local machine e.g.
  `ssh -N stat1001.eqiad.wmnet -L 8000:127.0.0.1:8000`
- Access the application and make calls from your local machine at http://localhost:8000
