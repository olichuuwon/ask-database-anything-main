# Use the official PostgreSQL image as the base image
FROM bitnami/postgresql:16.3.0-debian-12-r14

# Set environment variables for PostgreSQL
ENV POSTGRESQL_USERNAME=user
ENV POSTGRESQL_PASSWORD=pass
ENV POSTGRESQL_DATABASE=chinook

# Copy initialization scripts to the /tmp directory
COPY ./initdb/initdb.sql /tmp/

# RUN psql chinook -U user -w -f /tmp/initdb.sql

# Expose the PostgreSQL port
EXPOSE 5432

# Ensure permissions are correct for the initialization scripts
# RUN chown -R postgres:postgres /tmp

# Run the SQL file to initialize the database
RUN psql -U $POSTGRESQL_USERNAME -w -d $POSTGRESQL_DATABASE -f /tmp/initdb.sql
