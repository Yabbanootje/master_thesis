# master_thesis

## Build the image using

docker build -t image_name .

## Run the container using

docker run --mount type=bind,source="$(pwd)",target=/runs python-test
