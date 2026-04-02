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

HA_LANGUAGES = [
    "af", "ar", "bg", "bn", "ca", "cs", "cy", "da", "de_CH", "de", "el",
    "en", "en_US", "en_GB", "es", "et", "eu", "fa", "fi", "fr", "ga", "gl",
    "gu", "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ja", "ka", "kn",
    "ko", "kw", "lb", "lt", "lv", "ml", "mn", "mr", "ms", "nb", "ne", "nl",
    "pl", "pt_BR", "pt", "ro", "ru_RU", "sk", "sl", "sr", "sv", "sw",
    "ta", "te", "th", "tr", "uk", "ur", "vi", "zh_CN", "zh_HK", "zh_TW"
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
        help="Set a voice. Specify the path to a WAV file and its corresponding text. "
             "This argument can be used multiple times to add multiple voices.",
    )

    parser.add_argument("--uri", default="tcp://0.0.0.0:10204", help="URI of the server")
    parser.add_argument("--no-streaming", action="store_false", dest="streaming", help="Disable streaming")
    
    # --- OMNIVOICE PARAMETERS ---
    parser.add_argument("--language", default="auto", help="Language (e.g., 'ru', 'en', or 'auto' by default)")
    parser.add_argument("--guidance-scale", type=float, default=2.0, help="CFG Scale (default 2.0). Adjusts the accuracy of voice cloning.")
    parser.add_argument("--no-denoise", action="store_false", dest="denoise", help="Disable OmniVoice's built-in denoiser")
    parser.add_argument("--num-steps", type=int, default=16, help="Number of diffusion steps (default 16)")
    parser.add_argument("--speed", type=float, default=1.0, help="Synthesis speed (>1 faster, <1 slower)")
    
    parser.add_argument("--auto-punctuation", default=".?!", help="Auto-punctuation characters")
    parser.add_argument("--samples-per-chunk", type=int, default=1024, help="Samples per audio chunk")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--log-format", default=logging.BASIC_FORMAT, help="Log format")
    parser.add_argument("--version", action="version", version=__version__, help="Show version")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format)
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
    
    _LOGGER.info(f"Voices configured: {len(wyoming_voices)}. Language: {args.language}. Streaming: {'ON' if args.streaming else 'OFF'}.")
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