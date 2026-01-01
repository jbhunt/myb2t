# myb2t

## Overview
This is a repository for my submission to the [2025 Brain-to-Text Kaggle competition](https://www.kaggle.com/competitions/brain-to-text-25). The objective of this competition is to create a model to decode intended/attempted speech in patients with ALS or other severe speech disorders using intracranial recordings of neural activity and transcriptions of phoneme and character sequences. For my submission, I implemented a muli-task learning (MTL) encoder-decoder transformer that uses a character-level language model conditioned on the neural activity to predict target phoneme and character sequences. The language model component was pre-trained using a subset of the the English OPUS (OpenSubtitles) corpus.

## Datasets
For convenience, I've created PyTorch Dataset objects for the Brain-to-text (B2T) (`myb2t.datasets.BrainToText2025`) and OpenSubtitles (OPUS) datasets (`myb2t.datasets.OpusDataset`). The B2T dataset stores all sequences of neural activity (multi-unit spikes and LFPs) as well as sequences of character and phoneme tokens. The OPUS dataset contains only sequences of character tokens. Below is a visualization that illustrates the data available in each dataset:

![](data/imgs/datasets.png)

The first three subplots show sequences of character or phoneme tokens for 100 example sequences (y-axis). Color indicates the identity of the token (blue represents the pad token). The last two subplots show sequences of either multi-unit spiking (second-to-last) or LFP activity (far-right) across 256 recording channels (y-axis) for a single example trial. Color represents the magnitude of neural activity.
