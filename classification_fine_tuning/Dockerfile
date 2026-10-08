FROM python:3.11-slim

WORKDIR /app

COPY inference.py .
COPY checkpoint.pth .

COPY gptmodel/ ./gptmodel/
COPY tokenizer_bpe/ ./tokenizer_bpe/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tiktoken numpy

CMD ["python", "inference.py"]