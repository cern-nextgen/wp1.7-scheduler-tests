#!/bin/bash

# filepath: /home/cano/NGT/wp1.7-scheduler-tests/bin/simple_scheduler_scan.sh

# Determine the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Binary to execute (relative to the script's directory)
BINARY="$SCRIPT_DIR/schedule_simple"

# Number of events for each run
EVENTS=2000

# Output CSV header
echo "Slots,Threads,Bandwidth (events/sec)"

# Loop over threads (1 to 10)
for THREADS in {1..10}; do
    # Loop over slots (5 to 50, step 5)
    for SLOTS in $(seq 5 5 50); do
        # Run the binary and capture the output
        OUTPUT=$($BINARY --events $EVENTS --threads $THREADS --streams $SLOTS 2>&1)

        # Extract the bandwidth from the output
        BANDWIDTH=$(echo "$OUTPUT" | grep -oP '(?<=\().*?(?= events/sec)' | awk '{print $1}')

        # Print the result in CSV format
        echo "$SLOTS,$THREADS,$BANDWIDTH"
    done
done