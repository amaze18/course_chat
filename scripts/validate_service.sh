#!/bin/bash

# Define the URL to check
url="http://3.83.225.186:8080/"

# Check if Streamlit application is running
response=$(curl -s -o /dev/null -w "%{http_code}" $url)
if [[ "$response" == "200" ]]; then
    echo "Streamlit application is running."
else
    echo "Streamlit application is not running. HTTP response code: $response"
    echo "Debug info: URL=$url"
    # Adding additional checks for debugging
    echo "Checking if the server is reachable..."
    ping -c 4 3.83.225.186
    echo "Checking if the port is open..."
    nc -zv 3.83.225.186 8080
    exit 1
fi
