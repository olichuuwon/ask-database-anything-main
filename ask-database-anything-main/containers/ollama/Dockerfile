# Use the Ollama base image
FROM ollama/ollama:0.1.32

# Change file permissions
RUN mkdir /tmp/.ollama && \
    export OLLAMA_MODELS=/tmp/.ollama && \
    export OLLAMA_TMPDIR=/tmp/.ollama && \
    chgrp -R 0 /tmp/.ollama && \
    chmod -R g=u /tmp/.ollama 

# Pre-Install model
RUN bash -c "ollama serve &" && sleep 5 && ollama pull llama3:instruct

EXPOSE 11434
