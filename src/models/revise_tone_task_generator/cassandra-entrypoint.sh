#!/bin/bash
set -e

# Start Cassandra in the background
echo "Starting Cassandra..."
docker-entrypoint.sh cassandra -f &
CASSANDRA_PID=$!

# Wait for Cassandra to be ready
echo "Waiting for Cassandra to be ready..."
until cqlsh -e "describe cluster" > /dev/null 2>&1; do
  echo "Cassandra not ready yet, waiting..."
  sleep 5
done

echo "Cassandra is ready! Applying schema..."
cqlsh -f /schema.cql
echo "Schema applied successfully!"

# Bring Cassandra to the foreground
wait $CASSANDRA_PID
