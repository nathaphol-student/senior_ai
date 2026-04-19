# Kaggle Competition “หุ่นยนต์ TM หยิบยา”

### Team "Senior AI"

1. Tanut Apiwong, EGCO/M 6838839
2. Nathaphol Khingthong, EGCO/M 6838039
3. Nattapat Onkaew, EGCO/M 6836996

### How to run

1. Clone this git repository
2. `cd docker-images`
3. Build docker file with image name "medical-ai-service" and tag "latest"
4. In the `docker-compose.yml` file, the default device is `cpu`, if you have a CUDA device, please set to `cuda:0` or `0`
5. `docker compose up`
6. Open your browser, `http://localhost:8084/docs`
7. API endpoint for the Cobot: `http://localhost:8084/model` --> return an `x,y` coordinate
8. API endpoint for debugging: `http://localhost:8084/model_visualize` --> return image with detected bounding boxes and keypoints
