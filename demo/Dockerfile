FROM linuxcontainers/debian-slim:latest
WORKDIR /app
RUN apt update
RUN apt install -y python3 vim tmux curl git
# Clean up cache and temp files
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/*
# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN . $HOME/.local/bin/env
# install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh
# pull model
RUN ollama serve & sleep 5 ; \
    ollama pull llama3.2 ; \
    ps -ef | grep 'ollama serve' | grep -v grep | awk '{print $2}' | xargs -r kill -9
# install surrealdb
RUN curl -sSf https://install.surrealdb.com | sh