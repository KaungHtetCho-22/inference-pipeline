#!/bin/bash

LOG_FILE="logs/processing_log.txt"

echo "=====================================" >> "$LOG_FILE"
echo "$(date) - Starting Monitoring Service" >> "$LOG_FILE"
echo "=====================================" >> "$LOG_FILE"

nohup python3 watchdog.py >> "$LOG_FILE" 2>&1 &

echo "Watchdog service started."
