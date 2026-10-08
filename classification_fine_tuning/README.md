# SMS Spam Classifier

A lightweight **SMS spam/ham classifier** built with **PyTorch** and a GPT-2-style transformer architecture.

The application takes an SMS message from the terminal and classifies it as either:

* **ham** — a legitimate/non-spam message
* **spam** — a spam message

The project includes a custom BPE tokenizer, pretrained model weights, automatic device detection, and Docker support.

---

## Features

* GPT-2-style transformer architecture implemented with PyTorch
* Custom BPE tokenizer
* Binary SMS classification
* Pretrained model checkpoint
* Automatic hardware detection
* CUDA support when available
* Apple Silicon MPS support when available
* CPU fallback
* Automatic truncation of long inputs
* Interactive command-line interface
* Dockerized application
* Easy to download and run locally

---

## How It Works

The classifier follows this pipeline:

```text
SMS message
     │
     ▼
BPE Tokenizer
     │
     ▼
Token IDs
     │
     ▼
Context-length truncation
     │
     ▼
GPT-2-style Transformer
     │
     ▼
Classification Head
     │
     ▼
Prediction
     │
     ├── 0 → ham
     │
     └── 1 → spam
```

The model uses the final transformer output to predict one of two classes.

---

# Project Structure

The repository should have a structure similar to:

```text
sms-classifier/
│
├── README.md
├── Dockerfile
├── requirements.txt
├── inference.py
├── checkpoint.pth
│
├── gptmodel/
│   ├── __init__.py
│   ├── gpt_model.py
│   └── config.py
│   └── multiheadattention.py
└── tokenizer_bpe/
    ├── __init__.py
    └── __tokenizer.py
```

### Main files

| File                           | Purpose                                    |
| ------------------------------ | ------------------------------------------ |
| `inference.py`                 | Main application and inference entry point |
| `checkpoint.pth`               | Pretrained model weights                   |
| `requirements.txt`             | Python dependencies                        |
| `Dockerfile`                   | Docker image configuration                 |
| `gptmodel/gpt_model.py`        | GPT-style transformer architecture         |
| `gptmodel/config.py`           | Model configuration                        |
| `tokenizer_bpe/__tokenizer.py` | BPE tokenizer                              |
| `README.md`                    | Project documentation                      |

> Make sure the filenames in this README match the actual filenames in your repository.

---

# Requirements

You can run the project using either **Docker** or a local Python environment.

## Option 1 — Docker

Recommended.

You need:

* Docker
* Git

No Python installation is required on the host when using Docker.

## Option 2 — Local Python

You need:

* Python 3.9+
* pip
* PyTorch
* NumPy
* tiktoken

---

# Python Dependencies

The project uses the following dependencies:

```text
torch
tiktoken
numpy
```

They are specified in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

---

# Run with Docker

Docker is the recommended way to run the classifier.

## 1. Clone the repository


```bash
git clone https://huggingface.co/Skynet-G/gpt2_arch_sms_classifier/tree/main
```

Then enter the project directory:

```bash
cd gpt2_arch_sms_classifier
```

---

## 2. Build the Docker image

From the root of the project, run:

```bash
docker build -t sms-classifier .
```

This creates a Docker image named:

```text
sms-classifier
```

You only need to build the image again when the Dockerfile or application dependencies/code have changed.

---

## 3. Run the application

Start the container with:

```bash
docker run -it sms-classifier
```

The `-it` flags are required because the application uses interactive terminal input.

You should see:

```text
Type text to provide a text to the LLM
Type bye to terminate the program
```

---

# Using the Classifier

After starting the Docker container, type:

```text
text
```

The application will ask:

```text
Insert the text please:
```

Enter an SMS message.

For example:

```text
Insert the text please: Congratulations! You have won a free prize. Call now!
```

The classifier may return:

```text
spam
```

For a normal message:

```text
Insert the text please: Hey, are we still meeting at 6pm?
```

The classifier may return:

```text
ham
```

---

# Exit the Application

To terminate the application, type:

```text
bye
```

You will see:

```text
Goodbye
```

---

# Complete Docker Workflow

The complete workflow for a new user is:

```bash
git clone https://huggingface.co/Skynet-G/gpt2_arch_sms_classifier/tree/main

cd <REPOSITORY>

docker build -t sms-classifier .

docker run -it sms-classifier
```

Then:

```text
Type text to provide a text to the LLM
Type bye to terminate the program

text

Insert the text please: Congratulations! You won a free vacation!

spam
```

---

# Run Without Docker

Docker is not mandatory.

If you prefer to run the project directly with Python:

## 1. Clone the repository

```bash
git clone https://huggingface.co/Skynet-G/gpt2_arch_sms_classifier/tree/main
```

## 2. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the application

```bash
python inference.py
```

---

# Hardware Support

The application automatically detects the available device.

The detection order is:

```text
NVIDIA CUDA
     ↓
Apple Silicon MPS
     ↓
CPU
```

If a CUDA-compatible GPU is available, the application uses:

```text
cuda
```

If Apple Silicon MPS is available, it uses:

```text
mps
```

Otherwise, it falls back to:

```text
cpu
```

No manual device configuration is required for normal usage.

---

# Model

The project uses a GPT-2-style transformer model configured using:

```python
GPT_CONFIG_124M
```

The transformer output is adapted for binary classification using a two-class output head:

```python
torch.nn.Linear(
    in_features=cfg["emb_dim"],
    out_features=2
)
```

The classes are:

```text
0 → ham
1 → spam
```

The pretrained weights are loaded from:

```text
checkpoint.pth
```

The checkpoint is expected to contain:

```text
model_state_dict
```

For example:

```python
checkpoint = torch.load(
    "checkpoint.pth",
    map_location=self.device
)

self.gpt_model.load_state_dict(
    checkpoint["model_state_dict"]
)
```

---

# Tokenization

SMS messages are processed using a custom BPE tokenizer.

The input text is converted into token IDs:

```python
encoded_sms = self.tokenizer.encoder(sms)
```

If the resulting sequence is longer than the model's supported context length, it is truncated before being passed to the transformer.

This prevents inputs from exceeding the model's context window.

---

# Inference

The model performs inference without calculating gradients:

```python
with torch.no_grad():
    logits = self.gpt_model(input_ids)[:, -1, :]
```

The predicted class is selected using the highest logit:

```python
predicted_label = torch.argmax(
    logits,
    dim=-1
).item()
```

The result is then converted to:

```text
0 → ham
1 → spam
```

---

# Example Messages

### Spam

```text
Congratulations! You have won £1,000. Call now to claim your prize!
```

Classification:

```text
spam
```

### Ham

```text
Hey, are you free for dinner tonight?
```

Classification:

```text
ham
```

### Spam

```text
URGENT! You have been selected for a cash prize. Reply NOW!
```

Classification:

```text
spam
```

> These examples illustrate the intended behavior. Predictions are model-dependent and are not guaranteed to be correct.

---

# Dockerfile

The project can be built using the included `Dockerfile`.

A typical Dockerfile for this application is:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "inference.py"]
```

If your existing Dockerfile is different, keep your existing configuration and update this documentation to match it.

---

# Model Checkpoint

The application requires:

```text
checkpoint.pth
```

The checkpoint contains the pretrained model weights required for inference.

Make sure this file is available in the project when building the Docker image.

If the Dockerfile contains:

```dockerfile
COPY . .
```

the checkpoint will be copied into the Docker image along with the rest of the project.

### Large checkpoint files

If `checkpoint.pth` is large, consider using **Git LFS** or Hugging Face's large-file storage rather than committing a large binary directly to normal Git history.

---

# Hugging Face

This repository is designed to be downloadable from the Hugging Face Hub.

After downloading the repository, users can build and run the application locally:

```bash
git clone https://huggingface.co/Skynet-G/gpt2_arch_sms_classifier/tree/main
cd <gpt2_arch_sms_classifier>

docker build -t sms-classifier .

docker run -it sms-classifier
```

No external API is required for inference.

---

# Limitations

The current application is a terminal-based classifier.

It currently does not provide:

* Web interface
* REST API
* Batch CSV classification
* Python API
* Automatic model downloading
* Hugging Face Transformers `pipeline()`
* Confidence scores
* Browser-based inference

These could be added in future versions.

---

# Future Improvements

Possible improvements include:

* Add a FastAPI REST API
* Create a Hugging Face Space
* Add a web interface
* Add batch prediction
* Return classification probabilities
* Add model evaluation metrics
* Add automated tests
* Add CI/CD
* Publish a Docker image
* Convert/export the model to a standard Hugging Face format
* Add automatic checkpoint downloading

---

# Privacy

When running locally with Docker or Python, inference is performed locally using the model contained in the project.

SMS messages entered into the local application are not sent to an external inference API by this application.

Avoid entering sensitive information into environments you do not control.

---

# License


MIT License

# Author

**Georgios Mavropoulos**

GitHub: `https://www.linkedin.com/in/george-mavropoulos-509a6a326/`

Hugging Face: `https://huggingface.co/Skynet-G`
Email: `gmavropouloscoding@gmail.com`



