FROM nvcr.io/nvidia/base/ubuntu:jammy-20250415.1

# Install required OS dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy license information
RUN mkdir -p /legal
COPY LICENSE-3rd-party.txt /legal/

# Set working directory
WORKDIR /app

# Install Node.js LTS and pnpm
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g pnpm

# Print versions for debugging
RUN node -v && pnpm -v

# Environment variables
ARG NEXT_PUBLIC_MODEL_NAME
ARG VITE_EMBEDDING_MODEL
ARG VITE_RERANKER_MODEL
ARG VITE_API_CHAT_URL
ARG VITE_MODEL_NAME

ENV NEXT_PUBLIC_MODEL_NAME=${NEXT_PUBLIC_MODEL_NAME}
ENV VITE_EMBEDDING_MODEL=${VITE_EMBEDDING_MODEL}
ENV VITE_RERANKER_MODEL=${VITE_RERANKER_MODEL}
ENV VITE_API_CHAT_URL=${VITE_API_CHAT_URL}
ENV VITE_MODEL_NAME=${VITE_MODEL_NAME}

# Copy app files
COPY . .

# Change to frontend directory and install/build
WORKDIR /app/frontend

# Set environment variable to make pnpm non-interactive
ENV CI=true
ENV PNPM_HOME=/usr/local/bin
ENV npm_config_yes=true

RUN rm -rf ../node_modules ./node_modules && pnpm install --frozen-lockfile --reporter=silent
RUN pnpm add -D @types/node
RUN pnpm run build

# Expose and launch
EXPOSE 3000
CMD ["pnpm", "run", "preview"]
