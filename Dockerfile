# Build stage
FROM saiden/tengu:python-3.12-dev AS builder

WORKDIR /app

# Copy everything needed for the package
COPY . .

# Install dependencies and package
RUN uv pip install --system -e '.[server]'

# Runtime stage
FROM saiden/tengu:python-3.12

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app source
COPY --from=builder /app /app

# Set environment
ENV PATH="/usr/local/bin:$PATH"

EXPOSE 5000

CMD ["tsr", "serve", "--host", "0.0.0.0", "--port", "5000", "--sd-server", "http://172.17.0.1:1234"]
