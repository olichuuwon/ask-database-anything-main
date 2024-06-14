# Use the official PostgreSQL image as the base image
FROM bitnami/postgresql:16.3.0-debian-12-r14

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=user
ENV POSTGRES_PASSWORD=pass
ENV POSTGRES_DB=chinook

# Copy initialization scripts to the /tmp directory
COPY initdb /tmp/

# Expose the PostgreSQL port
EXPOSE 5432

# Ensure permissions are correct for the initialization scripts
RUN chown -R postgres:postgres /tmp

# Run the SQL file to initialize the database
RUN /opt/bitnami/postgresql/bin/psql -U $POSTGRES_USER -d $POSTGRES_DB -f /tmp/init.sql
