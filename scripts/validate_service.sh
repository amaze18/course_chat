#!/bin/bash

# Check if Streamlit application is running
response=$(curl -s -o /dev/null -w "%{http_code}" http://3.83.225.186:8080/)
if [[ "$response" == "200" ]]; then
    echo "Streamlit application is running."
else
    echo "Streamlit application is not running."
    exit 1
fi
