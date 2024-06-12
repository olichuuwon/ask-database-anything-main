# Use the official PostgreSQL image as the base image
FROM postgres:13

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=example_user
ENV POSTGRES_PASSWORD=example_password
ENV POSTGRES_DB=example_db

# Copy initialization scripts to the Docker entrypoint directory
COPY initdb /docker-entrypoint-initdb.d/

# Expose the PostgreSQL port
EXPOSE 5432
