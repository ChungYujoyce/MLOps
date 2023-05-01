<!-- ABOUT THE PROJECT -->
## About The Project
This is a project about MLOps.
Learning practical tools and approaches of developing a real product and go over the whole ML engineering (software engineering) process.

### Topics
- Building a simple model using huggingface pre-trained model.
- Model Monitoring - Weights and Bias
  - Using Weights & Bias to track all the experiments and tweak byper-parameters.
- Configuration management using Hydra
- Data Version Control - DVC 
  - Use DVC for model version control and upload saved model to Google Drive.
- Model Packaging - ONNX
  - Model formatting for different framework running.
- Model Packaging - Docker
  - Build docker container to run the app on different environment conveniently.
- CI/CD - GitHub Actions
  - Better developing workflow and deployment tracking for continuous build, test and deploy. 
- Container Registry - AWS ECR
  - Use a registry to store images created during the application development process.
- Serverless Deployment - AWS Lambda
  - Deploy Docker Image in ECR and triggering Lambda function with API Gateway. Automating deployment to Lambda using Github Actions.

### Result
API POST request from Postman and lambda function prediction returning.
![Postman](https://github.com/ChungYujoyce/MLOps/blob/master/postman_result.png)

### Credit
I followed this awesome [tutorial](https://www.ravirajag.dev/blog) step by step and learned a ton! 
