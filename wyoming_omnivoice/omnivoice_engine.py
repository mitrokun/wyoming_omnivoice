import os
import logging
import torch
import numpy as np

from omnivoice import OmniVoice, OmniVoiceGenerationConfig

_LOGGER = logging.getLogger(__name__)
MODEL_REPO = "k2-fsa/OmniVoice"

class OmniVoiceEngine:
    def __init__(self, voice_configs: dict, language: str, num_steps: int, speed: float, guidance_scale: float, denoise: bool):
        _LOGGER.info("Initializing OmniVoice engine...")
        self.speed = speed
        
        # Handle language (None triggers auto-detection in OmniVoice)
        self.language = None if language.lower() == "auto" else language

        # Create generation configuration based on generation-parameters.md
        self.gen_config = OmniVoiceGenerationConfig(
            num_step=num_steps,
            guidance_scale=guidance_scale,
            denoise=denoise,
            preprocess_prompt=True,   # Trims silence in reference audio
            postprocess_output=True   # Trims silence in generated output
        )

        # Device selection: CUDA > MPS > Error
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            raise RuntimeError("GPU (CUDA or MPS) not found. This server requires a GPU for inference.")
        
        self.model = self._load_tts_model()
        self.voice_references = {}

        _LOGGER.info("Checking reference audio files...")
        for name, config in voice_configs.items():
            _LOGGER.debug(f"Registering voice profile: {name}")
            ref_audio_path = config["ref_audio"]
            ref_text = config["ref_text"]
            
            if not os.path.exists(ref_audio_path):
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
            
            self.voice_references[name] = {
                "ref_audio": ref_audio_path,
                "ref_text": ref_text
            }
        
        _LOGGER.info(
            f"OmniVoice engine is ready. Device: {self.device}. "
            f"Profiles loaded: {len(self.voice_references)}"
        )

    def _load_tts_model(self):
        _LOGGER.info(f"Loading model '{MODEL_REPO}' from HF Hub...")
        try:
            model = OmniVoice.from_pretrained(
                MODEL_REPO,
                device_map=self.device,
                dtype=torch.float16
            )
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load OmniVoice model: {e}")

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        """
        Synthesizes speech from text using a specified voice profile.
        Returns a tuple of (audio_waveform, sample_rate).
        """
        if voice_name not in self.voice_references:
            fallback_name = next(iter(self.voice_references))
            _LOGGER.warning(f"Voice '{voice_name}' not found. Using fallback: '{fallback_name}'.")
            voice_name = fallback_name
            
        voice_config = self.voice_references[voice_name]

        # Generate audio tensors
        audio_tensors = self.model.generate(
            text=text,
            language=self.language,
            ref_audio=voice_config["ref_audio"],
            ref_text=voice_config["ref_text"],
            generation_config=self.gen_config,
            speed=self.speed
        )
        
        # Extract the first tensor and convert to NumPy array
        final_wave_tensor = audio_tensors[0].squeeze() 
        
        # Move to CPU if necessary before converting to NumPy
        if final_wave_tensor.is_cuda or final_wave_tensor.device.type == 'mps':
            final_wave_tensor = final_wave_tensor.cpu()
            
        final_wave = final_wave_tensor.numpy()
        
        # OmniVoice native sampling rate is 24kHz
        final_sample_rate = 24000
            
        return final_wave, final_sample_rate