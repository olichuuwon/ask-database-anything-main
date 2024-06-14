# Use the official PostgreSQL image as the base image
FROM bitnami/postgresql:16.3.0-debian-12-r14

# Set environment variables for PostgreSQL
ENV POSTGRESQL_USERNAME=user \
    POSTGRESQL_PASSWORD=pass \
    POSTGRESQL_DATABASE=chinook

# Copy initialization scripts to the /docker-entrypoint-initdb.d directory with appropriate ownership
COPY ./initdb/initdb.sql /docker-entrypoint-initdb.d/
COPY ./initdb/initdb.sql /tmp/

# Expose the PostgreSQL port
EXPOSE 5432
