# Use the Ollama base image
# FROM ollama/ollama:0.1.17
FROM quay.io/throwayaway13579/ollama-with-model

# RUN mkdir /.ollama

# RUN chgrp -R 0 / && \
#     chmod -R g=u /

# Copy the models into the appropriate directory
COPY ./data/.ollama/ /root/.ollama/

# Expose the port if you have a service running on a specific port
EXPOSE 11434
