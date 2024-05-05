#!/bin/bash



# Counter for numbering
counter=1

# Iterate over each file in the directory
for file in *.jpg *.jpeg *.png *.gif; do
    # Check if the file is a regular file
    if [ -f "$file" ]; then
        # Generate new filename with sequential numbering
        new_name="$counter.${file##*.}"  # Extract extension
        # Rename the file
        mv "$file" "$new_name"
        # Increment counter
        ((counter++))
    fi
done
