#!/bin/bash

# Start Streamlit application
nohup streamlit run test7.py --client.showErrorDetails=false --server.enableXsrfProtection=false --server.enableCORS=false --server.port=8080 &
echo -ne '\n'
