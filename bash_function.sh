#!/bin/bash

process_info() {
    echo
    echo "#############################"
    echo "$1 job started"
    echo "#############################"

    if [ -z "$1" ]; then
        echo "Process not found."
        exit 1
    fi

    # Start time of the script
    start_time=$(date +%s)

    # Loop to monitor the process
    while true; do
        
        if ps -p $1 > /dev/null; then
            # CPU usage
            cpu_usage=$(ps -p $1 -o %cpu | awk '{print 100 - $1"%"}')

            # RAM usage
            ram_usage=$(ps -p $1 -o %mem)

            # Calculate elapsed time
            elapsed_time=$(($(date +%s) - $start_time))

            # Convert seconds to minutes and seconds
            minutes=$((elapsed_time / 60))
            seconds=$((elapsed_time % 60))


            # Display the information
            echo
            echo "  Elapsed Time: ${minutes}m${seconds}s"
            echo "      CPU Usage: $cpu_usage"
            echo "      RAM Usage: $ram_usage"

            # Sleep for a bit before checking again
            sleep 5
        
        else
            break
        fi
    done

    echo
    echo "#############################"
    echo "$1 job completed."
    echo "#############################"
    echo
    echo 
}
