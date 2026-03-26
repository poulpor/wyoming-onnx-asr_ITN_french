# Wyoming Onnx ASR

**Disclaimer: Please note that the limited code contributed to this fork was largely developed with the assistance of Mistral’s Le Chat and Perplexity Pro conversational agents!**


Why the fork: 

I’ve noticed that models like parakeet-tdt-0.6b-v3 or Canary-1b-v2 do not apply the same Inverse Text Normalization (ITN) processing for English (en) as they do for multilingual settings. In English, numbers are transcribed as digits (e.g., "twenty one" → 21), whereas in multilingual mode—especially for French—numbers are transcribed as words (e.g., "vingt-et-un" → "vingt-et-un").

This creates an issue when using these models to control Home Assistant, which expects numbers for certain automations based on time (e.g., "12h30" instead of "douze heures trente"), volume settings (e.g., 20, 30...100%), etc.

This fork adds the "nemo-text-processing" package to the container, enabling post-processing of STT transcriptions. Once this package and its dependencies (such as gcc) are installed, post-processing can be added to the `handler.py` file.

Currently, this fork allows numbers to be transcribed as digits—for example, for times ("douze heures trente" → 12h30, "douze heure" → 12h00, etc.)—and can therefore be used with Home Assistant if this value is used in your automations.

You can add or modify the handler.py file as much as you like to include new rules or post-processing steps for the parakeet-tdt-0.6b-v3 and Canary-1b-v2 models (and likely others) currently managed by tbobby’s project. I warmly thank @tboby for their work, which has allowed me to use these high-performing models in French.



[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [onnx-asr](https://github.com/istupakov/onnx-asr/) speech to text system.

## Docker Compose:

Many thanks to [Leggz](https://www.reddit.com/r/homeassistant/comments/1r002uq/comment/o4f1gbj/) for the general layout.

```
services:
  wyoming_stt_nemo:
    image: poulpoche/wyoming-onnx-asr_itn_french
    container_name: wyoming-onnx-asr-itn-fr
    restart: unless-stopped
    ports:
      - "10300:10300"
    volumes:
      - /path/to/your/data/handler.py:/app/wyoming_onnx_asr/handler.py #path to your handler.py file for easier modification if needed (copy the one in this fork /wyoming_onnx_asr/ folder to apply french ITN for numbers)
      - /path/to/your/data:/data #path to the folder to store your chosen models
    command:
       - --uri 
       - "tcp://0.0.0.0:10300"
       - --model-multilingual
       # Choose one model between Parakeet and Canary, not both
       - "nemo-parakeet-tdt-0.6b-v3"
#      - "nemo-canary-1b-v2"
       - --model-dir
       - "/data"
       # Try changing "cpu" to "cuda" if you want to use the Orin AGX GPU
       - --device
       - "cpu"
```

    
The `--model-en` or `--model-multilingual` can also be a HuggingFace model but see [onnx-asr](https://github.com/istupakov/onnx-asr?tab=readme-ov-file#supported-model-names) for details

**NOTE**: Models are downloaded under `ONNX_ASR_MODEL_DIR` (default `/data` in Docker images), with a per-model subdirectory.
You may need to adjust this when using a read-only root filesystem (e.g., `ONNX_ASR_MODEL_DIR=/tmp`).
TensorRT engine cache remains under `/cache/tensorrt` when using the gpu-trt image.


## Configuration

- Quantization: the parakeet model supports int8, but make sure to compare as performance may or may not improve.
- Model cache directory: set `--model-dir` or `ONNX_ASR_MODEL_DIR` (default `/data`, per-model subdirectories).


## Running tooling
Install [mise](https://mise.jdx.dev/) and use `mise run` to get a list of tasks to test, format, lint, run.
