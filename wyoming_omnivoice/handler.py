import argparse
import asyncio
import io
import logging
import time
import wave
import re
import soundfile as sf

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .omnivoice_engine import OmniVoiceEngine
from .text_normalizer import TextNormalizer

_LOGGER = logging.getLogger(__name__)

class TTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        tts_engine: OmniVoiceEngine,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.tts_engine = tts_engine
        self.normalizer = TextNormalizer()
        
        self.min_chars = getattr(cli_args, "min_characters", 20)
        self.max_chars = getattr(cli_args, "max_characters", 200)

        self.sbd: SentenceBoundaryDetector | None = None
        self._synthesize: Synthesize | None = None
        self._is_streaming = False
        self._audio_started = False
        self._is_first_batch = True  # Переименовано для ясности
        
        self._sentence_buffer: str = ""
        _LOGGER.debug(f"Handler: Min chars: {self.min_chars}, Max chars: {self.max_chars}")

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            return True

        try:
            if self._is_streaming:
                if SynthesizeChunk.is_type(event.type):
                    await self._handle_stream_chunk(SynthesizeChunk.from_event(event))
                elif SynthesizeStop.is_type(event.type):
                    await self._handle_stream_stop()
                return True

            if SynthesizeStart.is_type(event.type) and self.cli_args.streaming:
                await self._handle_stream_start(SynthesizeStart.from_event(event))
            elif Synthesize.is_type(event.type):
                await self._handle_single_synthesize(Synthesize.from_event(event))
            
            return True

        except Exception as err:
            _LOGGER.exception("Error processing event")
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            self._is_streaming = False
        
        return True

    async def _handle_stream_start(self, stream_start: SynthesizeStart):
        self.sbd = SentenceBoundaryDetector()
        self._synthesize = Synthesize(text="", voice=stream_start.voice)
        self._is_streaming = True
        self._audio_started = False
        self._is_first_batch = True
        self._sentence_buffer = ""

    async def _process_sentence(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:
            return

        if self._sentence_buffer:
            self._sentence_buffer += " " + sentence
        else:
            self._sentence_buffer = sentence

        current_len = len(self._sentence_buffer)

        if self._is_first_batch:
            if current_len >= self.min_chars:
                _LOGGER.debug(f"First batch ready ({current_len} chars)")
                await self._flush_buffer()
                self._is_first_batch = False
        else:
            if current_len >= self.max_chars:
                _LOGGER.debug(f"Subsequent batch ready ({current_len} chars)")
                await self._flush_buffer()

    async def _flush_buffer(self):
        text_to_synthesize = self._sentence_buffer.strip()
        self._sentence_buffer = ""
        if text_to_synthesize:
            await self._synthesize_and_stream_audio(text_to_synthesize)

    async def _handle_stream_chunk(self, stream_chunk: SynthesizeChunk):
        assert self.sbd is not None
        for sentence in self.sbd.add_chunk(stream_chunk.text):
            await self._process_sentence(sentence)

    async def _handle_stream_stop(self):
        assert self.sbd is not None
        remaining_text = self.sbd.finish()
        if remaining_text:
            await self._process_sentence(remaining_text)
        await self._flush_buffer()

        if self._audio_started:
            await self.write_event(AudioStop().event())
        await self.write_event(SynthesizeStopped().event())
        self._is_streaming = False

    async def _handle_single_synthesize(self, synthesize: Synthesize):
        self._audio_started = False
        self._is_first_batch = True
        self._sentence_buffer = ""
        self._synthesize = synthesize 
        
        sbd = SentenceBoundaryDetector()
        sentences = list(sbd.add_chunk(synthesize.text))
        final_text = sbd.finish()
        if final_text:
            sentences.append(final_text)

        for sentence in sentences:
            await self._process_sentence(sentence)
        await self._flush_buffer()

        if self._audio_started:
            await self.write_event(AudioStop().event())

    async def _synthesize_and_stream_audio(self, text: str):
        if not self._synthesize or not self._synthesize.voice:
             return
        
        voice_name = self._synthesize.voice.name
        lang = self.cli_args.language.lower()
        
        if lang in ["ru", "rus", "russian", "ru-ru"] or (lang == "auto" and re.search(r'[а-яА-ЯёЁ]', text)):
            processed_text = self.normalizer.normalize(text)
        else:
            processed_text = text

        if not processed_text or not processed_text.strip():
            return

        if self.cli_args.auto_punctuation and processed_text[-1] not in self.cli_args.auto_punctuation:
            processed_text += self.cli_args.auto_punctuation[0]

        _LOGGER.debug("Synthesizing batch: '%s'", processed_text)
        start_time = time.monotonic()

        loop = asyncio.get_running_loop()
        final_wave, sample_rate = await loop.run_in_executor(
            None, self.tts_engine.synthesize, processed_text, voice_name
        )

        elapsed_time = time.monotonic() - start_time
        audio_duration = len(final_wave) / sample_rate
        rtfx = audio_duration / max(elapsed_time, 1e-6)

        _LOGGER.debug(f"Done: RTFX: {rtfx:.2f}x [{audio_duration:.2f}s audio / {elapsed_time:.2f}s calc]")
        
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, final_wave, sample_rate, format='WAV', subtype='PCM_16')
        wav_buffer.seek(0)
        
        with wave.open(wav_buffer, "rb") as wav_file:
            rate, width, channels = wav_file.getframerate(), wav_file.getsampwidth(), wav_file.getnchannels()
            
            if not self._audio_started:
                await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())
                self._audio_started = True
            
            audio_bytes = wav_file.readframes(wav_file.getnframes())
            bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
            
            for i in range(0, len(audio_bytes), bytes_per_chunk):
                chunk = audio_bytes[i : i + bytes_per_chunk]
                await self.write_event(AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event())