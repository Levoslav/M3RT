#!/bin/bash

cd ../..
ls

datasets=("marine" "photos")
labels=("short" "long")
models=("openclip")
openclip_versions=("ViT_SO400M_14_webli" "ViT_H_14_dfn5b" "ViT_L_16_webli" "ViT_B_16_webli" "ViT_G_14_laion2b" "ViT_H_14_laion2b")

# Loop over datasets
for dataset in "${datasets[@]}"; do
    # Loop over models
    for model in "${models[@]}"; do
        # Loop over label types
        for label in "${labels[@]}"; do
            # Print parameters
            echo "------------------------------------------------------------------"
            echo "Executing encode_images.py with parameters:"
            echo "  Dataset: $dataset"
            echo "  Labels: $label"
            echo "  Model: $model"

            # If model is "openclip", loop over versions
            if [ "$model" = "openclip" ]; then
                for version in "${openclip_versions[@]}"; do
                    # Print version
                    echo "  Version: $version"

                    # Execute encode_images.py with all parameters
                    python src/experiments/compute_ranks.py "$dataset" "$label" "$model" "$version"
                done
            else
                # Execute encode_images.py without version
                python src/experiments/compute_ranks.py "$dataset" "$label" "$model"
            fi

            echo "------------------------------------------------------------------"
        done
    done
done