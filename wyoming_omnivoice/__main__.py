import argparse
import asyncio
import logging
import os
from functools import partial

from wyoming.info import Attribution, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncServer

from . import __version__
from .omnivoice_engine import OmniVoiceEngine
from .handler import TTSEventHandler

_LOGGER = logging.getLogger(__name__)


class OmniVoiceColorFormatter(logging.Formatter):
    GREEN = "\033[38;2;38;162;105m"
    ORANGE = "\033[38;2;180;120;50m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    def format(self, record):
        log_message = super().format(record)
        msg = record.getMessage()
        
        if "Synth:" in msg:
            return f"{self.GREEN}{log_message}{self.RESET}"
        
        if "Done: RTFX:" in msg:
            return f"{self.ORANGE}{log_message}{self.RESET}"
        
        return f"{self.DIM}{log_message}{self.RESET}"

HA_LANGUAGES = [
    "af", "ar", "bg", "bn", "ca", "cs", "cy", "da", "de-CH", "de", "el",
    "en", "en-US", "en-GB", "es", "et", "eu", "fa", "fi", "fr", "ga", "gl",
    "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "ka", "kn",
    "ko", "kw", "lb", "lt", "lv", "ml", "mn", "mr", "ms", "nb", "ne", "nl",
    "pl", "pt-BR", "pt", "ro", "ru-RU", "sk", "sl", "sr", "sv", "sw", "ta",
    "te", "th", "tr", "uk", "ur", "vi", "zh-CN", "zh-HK", "zh-TW"
]

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--voice",
        required=True,
        nargs=2,
        action="append",
        metavar=("WAV_PATH", "TEXT"),
        help="Set a voice. Path to WAV file and its corresponding text.",
    )

    parser.add_argument("--uri", default="tcp://0.0.0.0:10204", help="URI of the server")
    parser.add_argument("--no-streaming", action="store_false", dest="streaming", help="Disable streaming")
    
    # --- OMNIVOICE PARAMETERS ---
    parser.add_argument("--language", default="auto", help="Language (e.g., 'ru', 'en')")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="CFG Scale")
    parser.add_argument("--no-denoise", action="store_false", dest="denoise", help="Disable denoiser")
    parser.add_argument("--num-steps", type=int, default=16, help="Number of diffusion steps")
    parser.add_argument("--speed", type=float, default=1.0, help="Synthesis speed")
    
    parser.add_argument(
        "--min-characters", 
        type=int, 
        default=20, 
        help="Min characters to buffer for the first synthesis request (default: 20)"
    )
    parser.add_argument(
        "--max-characters", 
        type=int, 
        default=200, 
        help="Max character limit for combining sentences after the first request (default: 200)"
    )
    
    parser.add_argument("--auto-punctuation", default=".?!", help="Auto-punctuation characters")
    parser.add_argument("--samples-per-chunk", type=int, default=1024, help="Samples per audio chunk")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT, help="Log format")
    parser.add_argument("--version", action="version", version=__version__, help="Show version")
    args = parser.parse_args()

    handler = logging.StreamHandler()
    handler.setFormatter(OmniVoiceColorFormatter(args.log_format))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    root_logger.addHandler(handler)

    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("omnivoice").setLevel(logging.INFO)
    logging.getLogger("wyoming").setLevel(logging.INFO)

    _LOGGER.debug(args)

    wyoming_voices = []
    engine_voice_configs = {}
    
    for i, voice_data in enumerate(args.voice):
        audio_path, text = voice_data
        stable_name = f"voice-{i+1:02d}"
        description = os.path.splitext(os.path.basename(audio_path))[0]
        
        wyoming_voices.append(
            TtsVoice(
                name=stable_name,
                description=description,
                attribution=Attribution(name="", url=""),
                installed=True,
                version=__version__,
                languages=HA_LANGUAGES,
            )
        )
        
        engine_voice_configs[stable_name] = {
            "ref_audio": audio_path,
            "ref_text": text,
        }

    wyoming_info = Info(
        tts=[
            TtsProgram(
                name="OmniVoice",
                description="Wyoming server for OmniVoice TTS",
                attribution=Attribution(name="OmniVoice", url="https://github.com/k2-fsa/OmniVoice"),
                installed=True,
                version=__version__,
                supports_synthesize_streaming=args.streaming,
                voices=wyoming_voices,
            )
        ],
    )
    
    _LOGGER.info(f"Voices configured: {len(wyoming_voices)}. Streaming: {'ON' if args.streaming else 'OFF'}.")
    _LOGGER.info("Initializing OmniVoice engine...")
    
    try:
        tts_engine = OmniVoiceEngine(
            voice_configs=engine_voice_configs,
            language=args.language,
            num_steps=args.num_steps,
            speed=args.speed,
            guidance_scale=args.guidance_scale,
            denoise=args.denoise
        )
    except RuntimeError as e:
        _LOGGER.fatal(e)
        return

    _LOGGER.info("Engine is ready.")

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Server is ready and listening on URI: %s", args.uri)
    
    await server.run(
        partial(
            TTSEventHandler,
            wyoming_info,
            args,
            tts_engine,
        )
    )

def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run()