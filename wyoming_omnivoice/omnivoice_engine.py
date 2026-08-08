import logging
import os

import numpy as np
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig, VoiceClonePrompt

_LOGGER = logging.getLogger(__name__)
MODEL_REPO = "k2-fsa/OmniVoice"


class OmniVoiceEngine:
    """Engine class for handling TTS synthesis using OmniVoice."""

    def __init__(
        self,
        voice_configs: dict,
        language: str,
        num_steps: int,
        speed: float,
        guidance_scale: float,
        denoise: bool,
    ) -> None:
        _LOGGER.info("Initializing OmniVoice engine...")
        self.speed = speed

        # Handle language (None triggers auto-detection in OmniVoice)
        self.language = None if language.lower() == "auto" else language

        # Create generation configuration based on model specs
        self.gen_config = OmniVoiceGenerationConfig(
            num_step=num_steps,
            guidance_scale=guidance_scale,
            denoise=denoise,
            preprocess_prompt=True,   # Trims silence in reference audio
            postprocess_output=True,  # Trims silence in generated output
        )

        # Device selection: CUDA > MPS > Error
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            raise RuntimeError(
                "GPU (CUDA or MPS) not found. This server requires a GPU for inference."
            )

        self.model = self._load_tts_model()
        self.voice_prompts: dict[str, VoiceClonePrompt] = {}

        # Initialize voice profiles and manage .pt cache files
        self._setup_and_manage_voice_cache(voice_configs)

        _LOGGER.info(
            "OmniVoice engine is ready. Device: %s. Profiles loaded: %d",
            self.device,
            len(self.voice_prompts),
        )

    def _setup_and_manage_voice_cache(self, voice_configs: dict) -> None:
        """
        Loads or creates a .pt prompt cache for each specified voice,
        and removes orphaned .pt files if their corresponding voices were removed.
        """
        _LOGGER.info("Checking voice reference files and prompt cache...")
        active_pt_paths: set[str] = set()
        checked_directories: set[str] = set()

        for name, config in voice_configs.items():
            ref_audio_path = os.path.abspath(config["ref_audio"])
            ref_text = config["ref_text"]

            if not os.path.exists(ref_audio_path):
                raise FileNotFoundError(
                    f"Reference audio file not found: {ref_audio_path}"
                )

            # Format cache file path (e.g., /path/to/voice.wav.pt)
            pt_cache_path = f"{ref_audio_path}.pt"
            active_pt_paths.add(pt_cache_path)
            checked_directories.add(os.path.dirname(ref_audio_path))

            # 1. Load from cache or create a new prompt
            if os.path.exists(pt_cache_path):
                _LOGGER.info(
                    "Loading cached voice prompt for '%s' from: %s",
                    name,
                    pt_cache_path,
                )
                try:
                    prompt = VoiceClonePrompt.load(pt_cache_path)
                except Exception as err:
                    _LOGGER.warning(
                        "Failed to load cached prompt '%s' (%s). Recreating...",
                        pt_cache_path,
                        err,
                    )
                    prompt = self._create_and_save_prompt(
                        ref_audio_path, ref_text, pt_cache_path
                    )
            else:
                _LOGGER.info(
                    "Creating new voice prompt for '%s' (%s)...",
                    name,
                    ref_audio_path,
                )
                prompt = self._create_and_save_prompt(
                    ref_audio_path, ref_text, pt_cache_path
                )

            self.voice_prompts[name] = prompt

        # 2. Cleanup orphaned .pt cache files
        self._cleanup_orphaned_cache(checked_directories, active_pt_paths)

    def _create_and_save_prompt(
        self, ref_audio_path: str, ref_text: str, pt_cache_path: str
    ) -> VoiceClonePrompt:
        """Creates a VoiceClonePrompt using the model and saves it to disk."""
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
        try:
            prompt.save(pt_cache_path)
            _LOGGER.info("Successfully saved voice prompt cache to: %s", pt_cache_path)
        except Exception as e:
            _LOGGER.error(
                "Failed to save voice prompt cache to '%s': %s", pt_cache_path, e
            )

        return prompt

    def _cleanup_orphaned_cache(
        self, directories: set[str], active_pt_paths: set[str]
    ) -> None:
        """Removes .pt files in voice directories if absent in the current run."""
        for directory in directories:
            if not os.path.exists(directory):
                continue

            for file_name in os.listdir(directory):
                if file_name.endswith(".pt"):
                    full_pt_path = os.path.abspath(os.path.join(directory, file_name))
                    if full_pt_path not in active_pt_paths:
                        _LOGGER.info(
                            "Removing orphaned voice prompt cache: %s", full_pt_path
                        )
                        try:
                            os.remove(full_pt_path)
                        except OSError as e:
                            _LOGGER.warning(
                                "Could not remove orphaned cache file '%s': %s",
                                full_pt_path,
                                e,
                            )

    def _load_tts_model(self) -> OmniVoice:
        """Loads OmniVoice model locally or downloads from HF Hub if missing."""
        _LOGGER.info("Loading model '%s' from HF Hub...", MODEL_REPO)
        try:
            _LOGGER.info("Trying to load model locally...")
            model = OmniVoice.from_pretrained(
                MODEL_REPO,
                device_map=self.device,
                dtype=torch.float16,
                local_files_only=True,
            )
            return model
        except Exception as local_err:
            _LOGGER.warning(
                "Local model not found (%s). Attempting to download...", local_err
            )
            try:
                model = OmniVoice.from_pretrained(
                    MODEL_REPO,
                    device_map=self.device,
                    dtype=torch.float16,
                    local_files_only=False,
                )
                return model
            except Exception as e:
                raise RuntimeError(f"Failed to load or download OmniVoice model: {e}")

    def synthesize(self, text: str, voice_name: str) -> tuple[np.ndarray, int]:
        """
        Synthesizes speech from text using a specified voice profile.
        Returns a tuple of (audio_waveform, sample_rate).
        """
        if voice_name not in self.voice_prompts:
            fallback_name = next(iter(self.voice_prompts))
            _LOGGER.warning(
                "Voice '%s' not found. Using fallback: '%s'.",
                voice_name,
                fallback_name,
            )
            voice_name = fallback_name

        prompt = self.voice_prompts[voice_name]

        # Pass pre-calculated voice_clone_prompt instead of ref_audio/ref_text
        audio_tensors = self.model.generate(
            text=text,
            language=self.language,
            voice_clone_prompt=prompt,
            generation_config=self.gen_config,
            speed=self.speed,
        )

        res = audio_tensors[0]

        if isinstance(res, torch.Tensor):
            final_wave = res.detach().cpu().numpy().squeeze()
        else:
            final_wave = np.array(res).squeeze()

        # OmniVoice native sampling rate is 24kHz
        final_sample_rate = 24000

        return final_wave, final_sample_rate