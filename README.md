# Language Identification with mT5 on GKE

### Introduction
This project deploys a fine-tuned mT5 (Multilingual Text-to-Text Transfer Transformer) model for language identification, hosted on Google Kubernetes Engine (GKE). The application efficiently identifies the language of the input text leveraging the powerful capabilities of mT5.

### Installation and Setup
##### Cloud Setup
1. Set up a Google Cloud account and enable billing.
2. Create a project and configure Google Kubernetes Engine within it.


### Local Environment Setup
1. Install Docker and Kubernetes CLI tools.
2. Configure local environment to interact with your Google Cloud account.

### Deployment Instructions
##### Kubernetes Deployment
1. Build the Docker image and push it to Docker Hub.
```
docker buildx build --platform linux/amd64 -t <image-name> .
docker push <image-name>
```
2. Use the provided Kubernetes manifest files to deploy the application to GKE.
```
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```
3. Enable port forwarding on GKE

##### Accessing kubectl logs
Access kubectl logs by using the following command
```
kubectl logs -f <pod-name>
```

##### Deploy application locally
The application uses gradio python package. To deploy locally, create a conda environment and install the requirements.
```
conda create --name myenv python=3.9
conda activate myenv
pip install -r requirements.txt
```
Place the weigths in the same folder as app.py and deploy the application using :
```
python3 app.py
```

### Fine-tuning mT5
The mT5 model was fine-tuned for language identification using a subset of the XNLI dataset (download from here). Details about the training process and dataset are available in the mT5_training.ipynb notebook.
Alternatively run the notebook directly in colab. 
<a target="_blank" href="https://colab.research.google.com/github/anishabhatnagar/CML-Final-Proj/blob/main/mT5_training.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Contributors
Anisha Bhatnagar(ab10945@nyu.edu)\
Divyanshi Parashar (dp3635@nyu.edu)\

### REFERENCES
1. https://github.com/KrishnanJothi/MT5_Language_identification_NLP
