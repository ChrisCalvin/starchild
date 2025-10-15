FROM hummingbot/hummingbot-api:latest

# Install netcat for healthcheck
RUN apt-get update && apt-get install -y netcat-traditional iputils-ping