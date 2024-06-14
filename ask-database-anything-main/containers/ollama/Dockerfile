# Use the Ollama base image
FROM ollama/ollama:0.1.32

# Set environment variables
ENV OLLAMA_MODELS=/tmp/.ollama/models
ENV OLLAMA_TMPDIR=/tmp/.ollama

# Set working directory
WORKDIR /tmp

# Change file permissions for models directory
RUN mkdir -p $OLLAMA_MODELS && \
    chgrp -R 0 $OLLAMA_MODELS && \
    chmod -R g=u $OLLAMA_MODELS

# Pre-Install model
RUN bash -c "ollama serve &" && sleep 5 && ollama pull llama3:instruct

# Ensure the /.ollama directory exists and copy id_ed25519 file
RUN mkdir -p /.ollama && \
    cp /root/.ollama/id_ed25519 /.ollama/id_ed25519

# Change permissions for the /.ollama directory
RUN chgrp -R 0 /.ollama && \
    chmod -R g=u /.ollama

# Change permissions for the /.ollama directory
RUN chgrp -R 0 /tmp && \
    chmod -R g=u /tmp

EXPOSE 11434
