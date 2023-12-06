# master_thesis

## Build the image

docker build -t image_name .

## Run the container

docker run --mount type=bind,source="$(pwd)",target=/app image_name

## Remove all images

docker system prune -a
