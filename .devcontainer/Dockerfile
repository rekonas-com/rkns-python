FROM python:3.11-slim

# Some utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# non-root user for security
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Set working directory
WORKDIR /workspaces/rkns

ENV POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # POETRY_VIRTUALENVS_CREATE=false \
    PATH="/opt/poetry/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

# ensure the poetry virtual environment is activated 
RUN echo 'source /workspaces/rkns/.venv/bin/activate || echo "Virtual environment not found. Run poetry install."' >> /home/$USERNAME/.bashrc


# Give ownership to the vscode user
RUN chown -R $USERNAME:$USERNAME /workspaces/rkns
USER $USERNAMEP
