#!/bin/bash

cd ../..
ls

# Define datasets, model names, and model versions
datasets=("marine" "photos")
models=("clip" "align" "blip2" "openclip")
openclip_versions=("ViT_SO400M_14_webli" "ViT_H_14_dfn5b" "ViT_L_16_webli" "ViT_B_16_webli" "ViT_G_14_laion2b" "ViT_H_14_laion2b")

# Loop over datasets
for dataset in "${datasets[@]}"; do
    # Loop over models
    for model in "${models[@]}"; do
        # Print parameters
        echo "------------------------------------------------------------------"
        echo "Executing encode_images.py with parameters:"
        echo "  Dataset: $dataset"
        echo "  Model: $model"

        # If model is "openclip", loop over versions
        if [ "$model" = "openclip" ]; then
            for version in "${openclip_versions[@]}"; do
                # Print version
                echo "  Version: $version"

                # Execute encode_images.py with all parameters
                python encode_images.py "$dataset" "$model" "$version"
            done
        else
            # Execute encode_images.py without version
            python encode_images.py "$dataset" "$model"
        fi

        echo "------------------------------------------------------------------"
    done
done
