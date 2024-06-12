# Use the official PostgreSQL image as the base image
FROM bitnami/postgresql:16.3.0-debian-12-r14

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=user
ENV POSTGRES_PASSWORD=pass
ENV POSTGRES_DB=chinook

# Copy initialization scripts to the Docker entrypoint directory
COPY initdb /docker-entrypoint-initdb.d/

# Expose the PostgreSQL port
EXPOSE 5432
