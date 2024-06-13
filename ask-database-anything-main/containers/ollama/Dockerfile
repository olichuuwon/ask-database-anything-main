# Use the Ollama base image
FROM ollama/ollama:0.1.32

# Expose the port if you have a service running on a specific port
EXPOSE 11434

# Pre-Install model
RUN bash -c "ollama serve &" && sleep 5 && ollama pull llama3:instruct
