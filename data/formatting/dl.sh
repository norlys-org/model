#!/bin/bash

# Assuming input file is 'urls.txt'
while read url; do
    # Get file name from URL
    file_name=$(basename "$url")

    # Download file
    wget "$url"

    # If the file is a tar file, uncompress it
    if [[ "$file_name" == *.tar ]]; then
        tar -xf "$file_name"
    fi
done < list.txt