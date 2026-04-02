# Wyoming OmniVoice

Prepare the environment according to the [documentation](https://github.com/k2-fsa/OmniVoice?tab=readme-ov-file#installation)

Install libs:
```
pip install omnivoice wyoming sentence-stream soundfile eng-to-ipa num2words
```
The last two libraries are used to enable the normalization block and are activated via the `--language` key with the value `ru`.

If the `--language` key is omitted, the model automatically detects the language (default behavior). You can also explicitly specify the language. This can improve quality, but it reduces multilingual support (especially when using different reference voices for different languages).


Run:
```
python -m wyoming_omnivoice --uri tcp://0.0.0.0:10204 --voice "C:\VS\olga.wav" "Reference text for Olga"

python3 -m wyoming_omnivoice --uri tcp://0.0.0.0:10204 \
  --voice "/home/user/voices/olga.wav" "Reference text for Olga"
```

You can specify any number of refs
