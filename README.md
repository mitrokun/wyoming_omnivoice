# Wyoming OmniVoice

Prepare the environment according to the [documentation](https://github.com/k2-fsa/OmniVoice?tab=readme-ov-file#installation)

Install libs:
```
pip install omnivoice wyoming sentence-stream soundfile eng-to-ipa num2words
```
The last two libraries are used only for the Russian language and are activated by the `--language ru` key.


Run:
```
python -m wyoming_omnivoice --uri tcp://0.0.0.0:10204 --voice "C:\VS\olga.wav" "Reference text for Olga"

python3 -m wyoming_omnivoice --uri tcp://0.0.0.0:10204 \
  --voice "/home/user/voices/olga.wav" "Reference text for Olga"
```

You can specify any number of refs
