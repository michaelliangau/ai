# Lao ASR

Improve the SOTA in Lao ASR and S2TT.

## Usage

Benchmark different ASR and S2TT providers
```bash
python benchmark.py --provider={provider_name} --model_task={model_task} --device={device}
```

## Benchmark Findings

I evaluated Whisper-v3-large, SeamlessStreaming and SeamlessM4T-v2-large on Lao speech to text translation and found that SeamlessM4T-v2-large is significantly better on understandability and fluency.

This is an example output from SeamlessM4T-v2-large:

```
Best prediction with 10-30s audio chunk splits along silences (total file length 1.5min)

Hello, my name is Mary. I was born in the village of Kankang in the province of Sidoun. I am married to a man of nineteen hundred and fifty-eight. I have eight children. There are six of us, two in Laos, four in the United States, and I thank God that he sent me and my son to a comfortable place. From the hardships of the world, to the joys of the world, to the joys of the world, to the joys of the world, to the joys of the world.
Come to my mother's house, she's fine, she's eating, she's eating, she's home, she's got a place to live. With the grace of God, he brought me here.The United States of America is a country that everyone wants to come to because it's the best place in the world. The U.S. government has taken care of the refugees who have come to live in our country. They take care of everything, young and old, everything is comfortable, and God has given them refuge in the best place in the world, the United States. Thank God for everything that has helped his people, all who have come to live in America, thank God for the Father.

Ground Truth
Hello, my name is Maly I was born in Laos	, Ban Khantherng, Pakse city, Sedong province. I got married in 1958. I have 8 children, 2 died, still have 6 remaining, 2 live in Laos, 4 live in the US. Firstly I want to thank God for blessing us with having a good life, bringing us comfort and preventing us from living in poverty. Living in the US is really comfortable, we have food on our plates and roof over our head. All of this is because of God’s blessing. Everyone wants to come to the US because it is the best place to live because the US government takes care of refugees, they take care of everything from food to accommodation because of God’s blessing we get to live in a blissful place on Earth. I thank God for everything for helping Lao people that seek refuge to the US and I thank Mother and Father.
```

Interestingly I found that how we split of the audio is incredibly important both because of GPU mem restrictions but also for output quality. It seems that SM4T might be sensitive to audio length or certain artefacts in some of the chunks. Generally speaking, I saw better translation performance at the start of files than the end and on shorter segments.

I also calculated quantitative BLEU metrics for the different providers where Whisper-v3-large wins out but a qualitative look at the results shows Whisper performs significantly worse semantically.

## Resources
- [Design Doc](https://docs.google.com/document/d/155EiskCDYdqvbzwpPG5GcXiEnxpWDsT0kloSUkkpOM4/edit?usp=sharing)