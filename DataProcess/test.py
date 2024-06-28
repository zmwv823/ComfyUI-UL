# Download the model and its components
# WhisperX is a system that includes multiple models for transcription, forced alignment, and diarization. For smooth SageMaker operation without the need to fetch model artifacts during inference, it’s essential to pre-download all model artifacts. These artifacts are then loaded into the SageMaker serving container during initiation. Because these models aren’t directly accessible, we offer descriptions and sample code from the WhisperX source, providing instructions on downloading the model and its components.

# WhisperX uses six models:

# A Faster Whisper model
# A Voice Activity Detection (VAD) model
# A Wav2Vec2 model
# pyannote’s Speaker Diarization model
# pyannote’s Segmentation model
# SpeechBrain’s Speaker Embedding model
# Most of these models can be obtained from Hugging Face using the huggingface_hub library. We use the following download_hf_model() function to retrieve these model artifacts. An access token from Hugging Face, generated after accepting the user agreements for the following pyannote models, is required:

# Speaker Diarization
# Segmentation
# Voice Activity Detection
import huggingface_hub
import yaml
import torchaudio
import urllib.request
import os

CONTAINER_MODEL_DIR = "/opt/ml/model"
WHISPERX_MODEL = "guillaumekln/faster-whisper-large-v2"
VAD_MODEL_URL = "https://whisperx.s3.eu-west-2.amazonaws.com/model_weights/segmentation/0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea/pytorch_model.bin"
WAV2VEC2_MODEL = "WAV2VEC2_ASR_BASE_960H"
DIARIZATION_MODEL = "pyannote/speaker-diarization"

def download_hf_model(model_name: str, hf_token: str, local_model_dir: str) -> str:
    """
    Fetches the provided model from HuggingFace and returns the subdirectory it is downloaded to
    :param model_name: HuggingFace model name (and an optional version, appended with @[version])
    :param hf_token: HuggingFace access token authorized to access the requested model
    :param local_model_dir: The local directory to download the model to
    :return: The subdirectory within local_modeL_dir that the model is downloaded to
    """
    model_subdir = model_name.split('@')[0]
    huggingface_hub.snapshot_download(model_subdir, token=hf_token, local_dir=f"{local_model_dir}/{model_subdir}", local_dir_use_symlinks=False)
    return model_subdir

# The VAD model is fetched from Amazon S3, and the Wav2Vec2 model is retrieved from the torchaudio.pipelines module. Based on the following code, we can retrieve all the models’ artifacts, including those from Hugging Face, and save them to the specified local model directory:
def fetch_models(hf_token: str, local_model_dir="./models"):
    """
    Fetches all required models to run WhisperX locally without downloading models every time 
    :param hf_token: A huggingface access token to download the models
    :param local_model_dir: The directory to download the models to
    """
    # Fetch Faster Whisper's Large V2 model from HuggingFace
    download_hf_model(model_name=WHISPERX_MODEL, hf_token=hf_token, local_model_dir=local_model_dir)

    # Fetch WhisperX's VAD Segmentation model from S3
    vad_model_dir = "whisperx/vad"
    if not os.path.exists(f"{local_model_dir}/{vad_model_dir}"):
        os.makedirs(f"{local_model_dir}/{vad_model_dir}")

    urllib.request.urlretrieve(VAD_MODEL_URL, f"{local_model_dir}/{vad_model_dir}/pytorch_model.bin")

    # Fetch the Wav2Vec2 alignment model
    torchaudio.pipelines.__dict__[WAV2VEC2_MODEL].get_model(dl_kwargs={"model_dir": f"{local_model_dir}/wav2vec2/"})

    # Fetch pyannote's Speaker Diarization model from HuggingFace
    download_hf_model(model_name=DIARIZATION_MODEL,
                      hf_token=hf_token,
                      local_model_dir=local_model_dir)

    # Read in the Speaker Diarization model config to fetch models and update with their local paths
    with open(f"{local_model_dir}/{DIARIZATION_MODEL}/config.yaml", 'r') as file:
        diarization_config = yaml.safe_load(file)

    embedding_model = diarization_config['pipeline']['params']['embedding']
    embedding_model_dir = download_hf_model(model_name=embedding_model,
                                            hf_token=hf_token,
                                            local_model_dir=local_model_dir)
    diarization_config['pipeline']['params']['embedding'] = f"{CONTAINER_MODEL_DIR}/{embedding_model_dir}"

    segmentation_model = diarization_config['pipeline']['params']['segmentation']
    segmentation_model_dir = download_hf_model(model_name=segmentation_model,
                                               hf_token=hf_token,
                                               local_model_dir=local_model_dir)
    diarization_config['pipeline']['params']['segmentation'] = f"{CONTAINER_MODEL_DIR}/{segmentation_model_dir}/pytorch_model.bin"

    with open(f"{local_model_dir}/{DIARIZATION_MODEL}/config.yaml", 'w') as file:
        yaml.safe_dump(diarization_config, file)

    # Read in the Speaker Embedding model config to update it with its local path
    speechbrain_hyperparams_path = f"{local_model_dir}/{embedding_model_dir}/hyperparams.yaml"
    with open(speechbrain_hyperparams_path, 'r') as file:
        speechbrain_hyperparams = file.read()

    speechbrain_hyperparams = speechbrain_hyperparams.replace(embedding_model_dir, f"{CONTAINER_MODEL_DIR}/{embedding_model_dir}")

    with open(speechbrain_hyperparams_path, 'w') as file:
        file.write(speechbrain_hyperparams)


# Create an inference script to load the models and run inference
# Next, we create a custom inference.py script to outline how the WhisperX model and its components are loaded into the container and how the inference process should be run. The script contains two functions: model_fn and transform_fn. The model_fn function is invoked to load the models from their respective locations. Subsequently, these models are passed to the transform_fn function during inference, where transcription, alignment, and diarization processes are performed. The following is a code sample for inference.py:        
import io
import json
import logging
import tempfile
import time

import torch
import whisperx

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def model_fn(model_dir: str) -> dict:
    """
    Deserialize and return the models
    """
    logging.info("Loading WhisperX model")
    model = whisperx.load_model(whisper_arch=f"{model_dir}/guillaumekln/faster-whisper-large-v2",
                                device=DEVICE,
                                language="en",
                                compute_type="float16",
                                vad_options={'model_fp': f"{model_dir}/whisperx/vad/pytorch_model.bin"})

    logging.info("Loading alignment model")
    align_model, metadata = whisperx.load_align_model(language_code="en",
                                                      device=DEVICE,
                                                      model_name="WAV2VEC2_ASR_BASE_960H",
                                                      model_dir=f"{model_dir}/wav2vec2")

    logging.info("Loading diarization model")
    diarization_model = whisperx.DiarizationPipeline(model_name=f"{model_dir}/pyannote/speaker-diarization/config.yaml",
                                                     device=DEVICE)

    return {
        'model': model,
        'align_model': align_model,
        'metadata': metadata,
        'diarization_model': diarization_model
    }

def transform_fn(model: dict, request_body: bytes, request_content_type: str, response_content_type="application/json") -> (str, str):
    """
    Load in audio from the request, transcribe and diarize, and return JSON output
    """

    # Start a timer so that we can log how long inference takes
    start_time = time.time()

    # Unpack the models
    whisperx_model = model['model']
    align_model = model['align_model']
    metadata = model['metadata']
    diarization_model = model['diarization_model']

    # Load the media file (the request_body as bytes) into a temporary file, then use WhisperX to load the audio from it
    logging.info("Loading audio")
    with io.BytesIO(request_body) as file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file.read())
        audio = whisperx.load_audio(tfile.name)

    # Run transcription
    logging.info("Transcribing audio")
    result = whisperx_model.transcribe(audio, batch_size=16)

    # Align the outputs for better timings
    logging.info("Aligning outputs")
    result = whisperx.align(result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False)

    # Run diarization
    logging.info("Running diarization")
    diarize_segments = diarization_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Calculate the time it took to perform the transcription and diarization
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Transcription and Diarization took {int(elapsed_time)} seconds")

    # Return the results to be stored in S3
    return json.dumps(result), response_content_type

# Within the model’s directory, alongside the requirements.txt file, ensure the presence of inference.py in a code subdirectory. The models directory should resemble the following:
# models
# ├── code
# │   ├── inference.py
# │   └── requirements.txt
# ├── guillaumekln
# │   └── faster-whisper-large-v2
# ├── pyannote
# │   ├── segmentation
# │   │   └── ...
# │   └── speaker-diarization
# │       └── ...
# ├── speechbrain
# │   └── spkrec-ecapa-voxceleb
# │       └── ...
# ├── wav2vec2
# │   └── ...
# └── whisperx
#     └── vad
#         └── ...