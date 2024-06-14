# Use the official PostgreSQL image as the base image
FROM bitnami/postgresql:16.3.0-debian-12-r14

# Set environment variables for PostgreSQL
ENV POSTGRESQL_USERNAME=user \
    POSTGRESQL_PASSWORD=pass \
    POSTGRESQL_DATABASE=chinook

# Copy initialization scripts to the /docker-entrypoint-initdb.d directory with appropriate ownership
COPY --chown=1001:0 ./initdb/initdb.sql /docker-entrypoint-initdb.d/

# Expose the PostgreSQL port
EXPOSE 5432
