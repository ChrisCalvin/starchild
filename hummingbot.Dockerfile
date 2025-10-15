# Use the official hummingbot image as a base
FROM hummingbot/hummingbot:latest

# Install the required python packages for our custom scripts
RUN pip install pandas paho-mqtt
