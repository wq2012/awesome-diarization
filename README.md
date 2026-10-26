# Awesome Speaker Diarization [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md)

## Table of contents

* [Overview](#overview)
* [Software](#software)
  * [Framework](#framework)
  * [Evaluation](#evaluation)
  * [Clustering](#clustering)
  * [Speaker embedding](#speaker-embedding)
  * [Speaker change detection](#speaker-change-detection)
  * [Audio feature extraction](#audio-feature-extraction)
  * [Audio data augmentation](#audio-data-augmentation)
  * [Other software](#other-software)
* [Datasets](#datasets)
  * [Diarization datasets](#diarization-datasets)
  * [Speaker embedding training sets](#speaker-embedding-training-sets)
  * [Augmentation noise sources](#augmentation-noise-sources)
* [Conferences](#conferences)
* [Products](#products)
* [Publications](#publications)
  * [Special topics](#special-topics)
    * [Review and survey papers](#review-and-survey-papers)
    * [Large language model (LLM)](#large-language-model-llm)
    * [Supervised diarization](#supervised-diarization)
    * [Joint diarization and ASR](#joint-diarization-and-asr)
    * [Online speaker diarization](#online-speaker-diarization)
    * [Challenges](#challenges)
    * [Audio-Visual Speaker Diarization](#audio-visual-speaker-diarization)
  * [Other](#other) ([2026](#2026) · [2025](#2025) · [2024](#2024) · [2023](#2023) · [2022](#2022) · [2021](#2021) · [2020](#2020) · [2019](#2019) · [2018](#2018) · [2017](#2017) · [2016](#2016) · [2015](#2015) · [2014](#2014) · [2013](#2013) · [2011](#2011) · [2009](#2009) · [2008](#2008) · [2006](#2006))
* [Other learning materials](#other-learning-materials)
  * [Online courses](#online-courses)
  * [Books](#books)
  * [Tech blogs](#tech-blogs)
  * [Video tutorials](#video-tutorials)
* [Star History](#star-history)

## Overview

This is a curated list of awesome Speaker Diarization papers, libraries, datasets, and other resources.

The purpose of this repo is to organize the world’s resources for speaker diarization, and make them universally accessible and useful.

To add items to this page, simply send a pull request. ([contributing guide](CONTRIBUTING.md))

## Software

### Framework

| Link | Language | Description |
| ---- | -------- | ----------- |
| [FunASR](https://github.com/modelscope/FunASR) ![GitHub stars](https://img.shields.io/github/stars/modelscope/FunASR?style=social) | Python & PyTorch | Industrial-grade speech recognition toolkit with built-in speaker diarization (cam++), VAD, ASR (SenseVoice/Paraformer), and punctuation. 50+ languages, 170x realtime. |
| [NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) ![GitHub stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Speech?style=social) | Python & PyTorch | Scalable generative AI and speech toolkit featuring end-to-end and modular speaker diarization (Sortformer, MSDD, TitaNet speaker embeddings, MarbleNet VAD, and joint ASR-diarization pipelines). |
| [WhisperX](https://github.com/m-bain/whisperX) ![GitHub stars](https://img.shields.io/github/stars/m-bain/whisperX?style=social) | Python & PyTorch | Fast automatic speech recognition with word-level timestamps (phoneme forced alignment) and speaker diarization via `pyannote-audio`. |
| [ESPnet](https://github.com/espnet/espnet) ![GitHub stars](https://img.shields.io/github/stars/espnet/espnet?style=social) | Python & PyTorch | End-to-end speech processing toolkit including neural speaker diarization (EEND, EEND-EDA) and joint speech enhancement/separation and diarization recipes. |
| [3D-Speaker](https://github.com/modelscope/3D-Speaker) ![GitHub stars](https://img.shields.io/github/stars/modelscope/3D-Speaker?style=social) | Python & PyTorch & C++ | Open-source toolkit for single- and multi-modal (audio-visual) speaker verification, recognition, and speaker diarization (CAM++, ERes2Net, overlap detection, and ONNX runtime). |
| [MiniVox](https://github.com/doerlbh/MiniVox) ![GitHub stars](https://img.shields.io/github/stars/doerlbh/MiniVox?style=social) | MATLAB | MiniVox is an open-source evaluation system for the online speaker diarization task. |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) ![GitHub stars](https://img.shields.io/github/stars/speechbrain/speechbrain?style=social) | Python & PyTorch | SpeechBrain is an open-source and all-in-one speech toolkit based on PyTorch. |
| [SIDEKIT for diarization (s4d)](https://lium.univ-lemans.fr/en/s4d/) | Python | An open-source package extension of SIDEKIT for speaker diarization ([source](https://git-lium.univ-lemans.fr/Meignier/s4d)). |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |
| [AaltoASR](https://github.com/aalto-speech/speaker-diarization) ![GitHub stars](https://img.shields.io/github/stars/aalto-speech/speaker-diarization?style=social) | Python & Perl | Speaker diarization scripts, based on AaltoASR. |
| [LIUM SpkDiarization](https://git-lium.univ-lemans.fr/Meignier/lium-spkdiarization) | Java | LIUM_SpkDiarization is software dedicated to speaker diarization (i.e. speaker segmentation and clustering). It is written in Java, and includes developments in the domain (as of 2013). |
| [kaldi-asr](https://github.com/kaldi-asr/kaldi/tree/master/egs/callhome_diarization) ![GitHub stars](https://img.shields.io/github/stars/kaldi-asr/kaldi?style=social) | Bash | Example scripts for speaker diarization on a portion of CALLHOME used in the 2000 NIST speaker recognition evaluation. |
| [kaldi-speaker-diarization](https://github.com/cadia-lvl/kaldi-speaker-diarization) ![GitHub stars](https://img.shields.io/github/stars/cadia-lvl/kaldi-speaker-diarization?style=social) | Bash | Icelandic speaker diarization scripts using kaldi. |
| [Alize LIA_SpkSeg](https://github.com/ALIZE-Speaker-Recognition/LIA_RAL) ![GitHub stars](https://img.shields.io/github/stars/ALIZE-Speaker-Recognition/LIA_RAL?style=social) | C++ | ALIZÉ is an open-source platform for speaker recognition; LIA_SpkSeg provides the tools for speaker diarization. |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-audio?style=social) | Python & PyTorch | Neural building blocks for speaker diarization: speech activity detection, speaker change detection, speaker embedding. |
| [pyBK](https://github.com/josepatino/pyBK) ![GitHub stars](https://img.shields.io/github/stars/josepatino/pyBK?style=social) | Python | Speaker diarization using binary key speaker modelling. Computationally light solution that does not require external training data. |
| [Speaker-Diarization](https://github.com/taylorlu/Speaker-Diarization) ![GitHub stars](https://img.shields.io/github/stars/taylorlu/Speaker-Diarization?style=social) | Python | Speaker diarization using uis-rnn and GhostVLAD. An easier way to support openset speakers. |
| [EEND](https://github.com/hitachi-speech/EEND) ![GitHub stars](https://img.shields.io/github/stars/hitachi-speech/EEND?style=social) | Python & Bash & Perl | End-to-End Neural Diarization. |
| [VBx](https://github.com/BUTSpeechFIT/VBx) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/VBx?style=social) | Python | Variational Bayes HMM over x-vectors diarization. x-vector extractor [recipe](https://github.com/phonexiaresearch/VBx-training-recipe) |
| [RE-VERB](https://github.com/team-re-verb/RE-VERB) ![GitHub stars](https://img.shields.io/github/stars/team-re-verb/RE-VERB?style=social) | Python & JavaScript | RE: VERB is speaker diarization system, it allows the user to send/record audio of a conversation and receive timestamps of who spoke when. |
| [diart](https://github.com/juanmc2005/diart) ![GitHub stars](https://img.shields.io/github/stars/juanmc2005/diart?style=social) | Python & PyTorch | Real-time/streaming speaker diarization extending [pyannote.audio](https://github.com/pyannote/pyannote-audio) to online processing (formerly `StreamingSpeakerDiarization`). |
| [simple_diarizer](https://github.com/cvqluu/simple_diarizer) ![GitHub stars](https://img.shields.io/github/stars/cvqluu/simple_diarizer?style=social) | Python | Simplified diarization pipeline using pretrained models. Made to be as simple as possible to go from an input audio file to diarized segments. |
| [Picovoice Falcon](https://github.com/Picovoice/falcon) ![GitHub stars](https://img.shields.io/github/stars/Picovoice/falcon?style=social) | C & Python | A [lightweight, accurate, and fast](https://picovoice.ai/docs/benchmark/speaker-diarization/#accuracy) speaker diarization engine written in C and available in Python, running on CPU with minimal overhead. |
| [DiaPer](https://github.com/BUTSpeechFIT/DiaPer) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/DiaPer?style=social) | Python & PyTorch | PyTorch implementation for [DiaPer: End-to-End Neural Diarization with Perceiver-Based Attractors](https://arxiv.org/abs/2312.04324) including models pre-trained on free and public data. |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) ![GitHub stars](https://img.shields.io/github/stars/k2-fsa/sherpa-onnx?style=social) | C++ & C & `C#` & Dart & Go & Java & JavaScript & Kotlin & Pascal & Python & Rust & Swift | Supports speaker diarization, speech recognition, and text-to-speech on various platforms with various language bindings. |
| [FluidAudio](https://github.com/FluidInference/FluidAudio) ![GitHub stars](https://img.shields.io/github/stars/FluidInference/FluidAudio?style=social) | Swift | A native Swift speaker diarization library for Apple platforms, using CoreML for efficient, real-time audio processing with high accuracy. |
| [speakrs](https://github.com/avencera/speakrs) ![GitHub stars](https://img.shields.io/github/stars/avencera/speakrs?style=social) | Rust & Python | Rust implementation of the pyannote community-1 diarization pipeline (segmentation, embedding, PLDA, VBx clustering) with native CPU, CUDA, and CoreML backends. 7.0% DER on VoxConverse dev at 1912x realtime on an RTX 4090. |
| [diarizers](https://github.com/diarizers-community/diarizers) ![GitHub stars](https://img.shields.io/github/stars/diarizers-community/diarizers?style=social) [![Build Status](https://github.com/diarizers-community/diarizers/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/diarizers-community/diarizers/actions/workflows/pythonapp.yml) | Python & PyTorch | Unified pipeline for ASR, standalone speaker diarization (neural segmentation, ECAPA-TDNN speaker embeddings, spectral clustering), and DiarizationLM post-processing, with an interactive [Hugging Face Space](https://huggingface.co/spaces/diarizers-community/diarizers). |


### Evaluation

| Link | Language | Description |
| ---- | -------- | ----------- |
| [pyannote-metrics](https://github.com/pyannote/pyannote-metrics) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-metrics?style=social) | Python | A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems. |
| [SimpleDER](https://github.com/wq2012/SimpleDER) ![GitHub stars](https://img.shields.io/github/stars/wq2012/SimpleDER?style=social) [![Build Status](https://github.com/wq2012/SimpleDER/actions/workflows/python-package.yml/badge.svg)](https://github.com/wq2012/SimpleDER/actions/workflows/python-package.yml) | Python | A lightweight library to compute Diarization Error Rate (DER), Jaccard Error Rate (JER), and Word Diarization Error Rate (WDER). |
| [DiarizationLM](https://github.com/google/speaker-id/blob/master/DiarizationLM/README.md) ![GitHub stars](https://img.shields.io/github/stars/google/speaker-id?style=social) [![Build Status](https://github.com/google/speaker-id/actions/workflows/python-app-diarizationlm.yml/badge.svg)](https://github.com/google/speaker-id/actions/workflows/python-app-diarizationlm.yml) | Python | Implements Word Error Rate (WER), Word Diarization Error Rate (WDER), and concatenated minimum-permutation Word Error Rate (cpWER). |
| NIST md-eval | Perl | (1) [md-eval-v21.pl](https://github.com/jitendrab/btp/blob/master/c_code/single_diag_gaussian_no_viterbi/md-eval-v21.pl) from [jitendra](https://github.com/jitendrab); (2) [md-eval-22.pl](https://github.com/nryant/dscore/blob/master/scorelib/md-eval-22.pl) from [nryant](https://github.com/nryant) |
| [dscore](https://github.com/nryant/dscore) ![GitHub stars](https://img.shields.io/github/stars/nryant/dscore?style=social) | Python & Perl | Diarization scoring tools. |
| [Sequence Match Accuracy](https://github.com/google/uis-rnn/blob/master/uisrnn/evals.py) ![GitHub stars](https://img.shields.io/github/stars/google/uis-rnn?style=social) | Python | Match the accuracy of two sequences with Hungarian algorithm. |
| [spyder](https://github.com/desh2608/spyder) ![GitHub stars](https://img.shields.io/github/stars/desh2608/spyder?style=social) | Python & C++ | Simple Python package for fast DER computation. |
| [CDER](https://github.com/SpeechClub/CDER_Metric) ![GitHub stars](https://img.shields.io/github/stars/SpeechClub/CDER_Metric?style=social) | Python | Conversational DER from [The Conversational Short-phrase Speaker Diarization (CSSD) Task: Dataset, Evaluation Metric and Baselines](https://arxiv.org/abs/2208.08042) |
| [Selective Collar](https://github.com/shahmananv13/selective_collar) ![GitHub stars](https://img.shields.io/github/stars/shahmananv13/selective_collar?style=social) | Python | Selective collar for mitigating evaluation bias in speaker diarization from [Beyond Uniform Forgiveness: Introducing the Selective Collar to Mitigate Evaluation Bias in Speaker Diarization](https://doi.org/10.1109/ACCESS.2026.3729661) |

### Clustering

| Link | Language | Description |
| ---- | -------- | ----------- |
| [uis-rnn](https://github.com/google/uis-rnn) ![GitHub stars](https://img.shields.io/github/stars/google/uis-rnn?style=social) | Python & PyTorch | Google's Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN) algorithm, for Fully Supervised Speaker Diarization. This clustering algorithm is **supervised**. |
| [uis-rnn-sml](https://github.com/DonkeyShot21/uis-rnn-sml) ![GitHub stars](https://img.shields.io/github/stars/DonkeyShot21/uis-rnn-sml?style=social) | Python & PyTorch | A variant of UIS-RNN, for the paper Supervised Online Diarization with Sample Mean Loss for Multi-Domain Data. |
| [DNC](https://github.com/FlorianKrey/DNC) ![GitHub stars](https://img.shields.io/github/stars/FlorianKrey/DNC?style=social) | Python & ESPnet | Transformer-based Discriminative Neural Clustering (DNC) for Speaker Diarisation. Like UIS-RNN, it is **supervised**. |
| [SpectralCluster](https://github.com/wq2012/SpectralCluster) ![GitHub stars](https://img.shields.io/github/stars/wq2012/SpectralCluster?style=social) | Python | Spectral clustering with affinity matrix refinement operations, auto-tune, and speaker turn constraints. |
| [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html) | Python | scikit-learn clustering algorithms. |
| [PLDA](https://github.com/RaviSoji/plda) ![GitHub stars](https://img.shields.io/github/stars/RaviSoji/plda?style=social) | Python | Probabilistic Linear Discriminant Analysis & classification, written in Python. |
| [PLDA](https://github.com/mrouvier/plda) ![GitHub stars](https://img.shields.io/github/stars/mrouvier/plda?style=social) | C++ | Open-source implementation of simplified PLDA (Probabilistic Linear Discriminant Analysis). |
| [Auto-Tuning Spectral Clustering](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering) ![GitHub stars](https://img.shields.io/github/stars/tango4j/Auto-Tuning-Spectral-Clustering?style=social) | Python | Auto-tuning Spectral Clustering method that does not need development set or supervised tuning. |


### Speaker embedding

| Link | Method | Language | Description |
| ---- | ------ | -------- | ----------- |
| [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) ![GitHub stars](https://img.shields.io/github/stars/resemble-ai/Resemblyzer?style=social) | d-vector | Python & PyTorch | PyTorch implementation of generalized end-to-end loss for speaker verification, which can be used for voice cloning and diarization. |
| [Speaker_Verification](https://github.com/Janghyun1230/Speaker_Verification) ![GitHub stars](https://img.shields.io/github/stars/Janghyun1230/Speaker_Verification?style=social) | d-vector | Python & TensorFlow | TensorFlow implementation of generalized end-to-end loss for speaker verification. |
| [PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification) ![GitHub stars](https://img.shields.io/github/stars/HarryVolek/PyTorch_Speaker_Verification?style=social) | d-vector | Python & PyTorch | PyTorch implementation of "Generalized End-to-End Loss for Speaker Verification" by Wan, Li et al. With UIS-RNN integration. |
| [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) ![GitHub stars](https://img.shields.io/github/stars/CorentinJ/Real-Time-Voice-Cloning?style=social) | d-vector | Python & PyTorch | Implementation of "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis" (SV2TTS) with a vocoder that works in real-time. |
| [conformer-speaker-encoder](https://huggingface.co/tflite-hub/conformer-speaker-encoder) | d-vector | Python & TFLite | Massively multilingual conformer-based speaker recognition models in TFLite format. |
| [deep-speaker](https://github.com/philipperemy/deep-speaker) ![GitHub stars](https://img.shields.io/github/stars/philipperemy/deep-speaker?style=social) | d-vector | Python & Keras | Third party implementation of the Baidu paper Deep Speaker: an End-to-End Neural Speaker Embedding System. |
| [x-vector-kaldi-tf](https://github.com/BUTSpeechFIT/x-vector-kaldi-tf) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/x-vector-kaldi-tf?style=social) | x-vector | Python & TensorFlow & Perl | TensorFlow implementation of x-vector topology on top of Kaldi recipe. |
| [kaldi-ivector](https://github.com/idiap/kaldi-ivector) ![GitHub stars](https://img.shields.io/github/stars/idiap/kaldi-ivector?style=social) | i-vector | C++ & Perl | Extension to Kaldi implementing the standard i-vector hyperparameter estimation and i-vector extraction procedure. |
| [voxceleb-ivector](https://github.com/swshon/voxceleb-ivector) ![GitHub stars](https://img.shields.io/github/stars/swshon/voxceleb-ivector?style=social) | i-vector | Perl | Voxceleb1 i-vector based speaker recognition system. |
| [pytorch_xvectors](https://github.com/manojpamk/pytorch_xvectors) ![GitHub stars](https://img.shields.io/github/stars/manojpamk/pytorch_xvectors?style=social) | x-vector | Python & PyTorch | PyTorch implementation of Voxceleb x-vectors. Additionally, includes meta-learning architectures for embedding training. Evaluated with speaker diarization and speaker verification. |
| [ASVtorch](https://gitlab.com/ville.vestman/asvtorch) | i-vector | Python & PyTorch | ASVtorch is a toolkit for automatic speaker recognition. |
| [asv-subtools](https://github.com/Snowdar/asv-subtools) ![GitHub stars](https://img.shields.io/github/stars/Snowdar/asv-subtools?style=social) | i-vector & x-vector | Kaldi & PyTorch | ASV-Subtools is developed based on PyTorch and Kaldi for the task of speaker recognition, language identification, etc. The 'sub' of 'subtools' means that there are many modular tools and the parts constitute the whole. |
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker) ![GitHub stars](https://img.shields.io/github/stars/wenet-e2e/wespeaker?style=social) | x-vector & r-vector | Python & C++ & PyTorch | WeSpeaker is a research and production oriented speaker verification, recognition and diarization toolkit, which supports very strong recipes with on-the-fly data preparation, model training and evaluation, as well as runtime C++ codes. |
| [ReDimNet](https://github.com/IDRnD/redimnet) ![GitHub stars](https://img.shields.io/github/stars/IDRnD/redimnet?style=social) | improved resnet | Python & PyTorch | Neural network architecture presented in the paper [Reshape Dimensions Network for Speaker Recognition](https://arxiv.org/abs/2407.18223) |


### Speaker change detection

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [change_detection](https://github.com/yinruiqing/change_detection) ![GitHub stars](https://img.shields.io/github/stars/yinruiqing/change_detection?style=social) | Python & Keras | Code for Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks. |
| [tinydiarize](https://github.com/akashmjn/tinydiarize) ![GitHub stars](https://img.shields.io/github/stars/akashmjn/tinydiarize?style=social) | Python & PyTorch | Diarization inside OpenAI Whisper decoder |

### Audio feature extraction

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [LibROSA](https://github.com/librosa/librosa) ![GitHub stars](https://img.shields.io/github/stars/librosa/librosa?style=social) | Python | Python library for audio and music analysis. https://librosa.github.io/ |
| [python_speech_features](https://github.com/jameslyons/python_speech_features) ![GitHub stars](https://img.shields.io/github/stars/jameslyons/python_speech_features?style=social) | Python | This library provides common speech features for ASR including MFCCs and filterbank energies. https://python-speech-features.readthedocs.io/en/latest/ |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |

### Audio data augmentation

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) ![GitHub stars](https://img.shields.io/github/stars/LCAV/pyroomacoustics?style=social) | Python | Pyroomacoustics is a package for audio signal processing for indoor applications. It was developed as a fast prototyping platform for beamforming algorithms in indoor scenarios. https://pyroomacoustics.readthedocs.io |
| [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR) ![GitHub stars](https://img.shields.io/github/stars/DavidDiazGuerra/gpuRIR?style=social) | Python | Python library for Room Impulse Response (RIR) simulation with GPU acceleration |
| [rir_simulator_python](https://github.com/sunits/rir_simulator_python) ![GitHub stars](https://img.shields.io/github/stars/sunits/rir_simulator_python?style=social) | Python | Room impulse response simulator using python |
| [WavAugment](https://github.com/facebookresearch/WavAugment) ![GitHub stars](https://img.shields.io/github/stars/facebookresearch/WavAugment?style=social) | Python & PyTorch | WavAugment performs data augmentation on audio data. The audio data is represented as PyTorch tensors. |
| [EEND_dataprep](https://github.com/BUTSpeechFIT/EEND_dataprep) ![GitHub stars](https://img.shields.io/github/stars/BUTSpeechFIT/EEND_dataprep?style=social) | Bash & Python | Recipes for generating [simulated conversations](https://arxiv.org/abs/2204.00890) used to train end-to-end diarization models. |

### Other software

| Link | Language | Description |
| ---- | -------- | ----------- |
| [VB Diarization](https://github.com/wq2012/VB_diarization) ![GitHub stars](https://img.shields.io/github/stars/wq2012/VB_diarization?style=social) | Python | VB Diarization with Eigenvoice and HMM Priors. |
| [DOVER-Lap](https://github.com/desh2608/dover-lap) ![GitHub stars](https://img.shields.io/github/stars/desh2608/dover-lap?style=social) | Python | Python package for combining diarization system outputs |
| [Diar-az](https://github.com/cadia-lvl/diar-az) ![GitHub stars](https://img.shields.io/github/stars/cadia-lvl/diar-az?style=social) | Bash & Python | Data formatting tool to support the ruv-di dataset. Kaldi to Gecko to Kaldi and corpus and back. |
| [voxsolo](https://github.com/dhayanithi-svaromance/voxsolo) ![GitHub stars](https://img.shields.io/github/stars/dhayanithi-svaromance/voxsolo?style=social) | Python | Verbatim per-speaker audio from diarization: keeps one speaker bit-exact, silences other speakers and overlapped speech; exports EDL/SRT/VTT/Audacity labels for editors. |
| [VoxLock](https://github.com/arbazsiddiqui/voxlock) ![GitHub stars](https://img.shields.io/github/stars/arbazsiddiqui/voxlock?style=social) | Python | Streaming target speaker extraction: keeps one speaker's voice and removes the other voices in the room before speech recognition. 27M parameters, 24 ms algorithmic latency on CPU, ONNX and safetensors weights. |
| [OATS](https://github.com/ariso-ai/oats) ![GitHub stars](https://img.shields.io/github/stars/ariso-ai/oats?style=social) | Rust & TypeScript | Open-source macOS app for meeting transcription and notes with speaker diarization (FluidAudio) and optional fully offline mode. |
| [akou](https://github.com/GeiserX/akou) ![GitHub stars](https://img.shields.io/github/stars/GeiserX/akou?style=social) | TypeScript & Rust | Open-source macOS app that records calls and meetings and transcribes them on-device, with speaker labels from NVIDIA Nemotron 3 Diarization. |
| [Local Transcriber](https://github.com/teconomix/local-transcriber) ![GitHub stars](https://img.shields.io/github/stars/teconomix/local-transcriber?style=social) | Python & Swift | macOS menu bar app that records calls and meetings and transcribes them on-device (Whisper large-v3-turbo on MLX), with pyannote diarization per audio track and a voice database that recognizes known speakers. |

## Datasets

### Diarization datasets

| Audio | Diarization ground truth | Language | Pricing | Additional information |
| ----- | ------------------------ | -------- | ------- | ---------------------- |
| [2000 NIST Speaker Recognition Evaluation](https://catalog.ldc.upenn.edu/LDC2001S97) | [Disk-6 (Switchboard)](https://github.com/google/speaker-id/tree/master/publications/LstmDiarization/evaluation/NIST_SRE2000/Disk6_ground_truth), [Disk-8 (CALLHOME)](https://github.com/google/speaker-id/tree/master/publications/LstmDiarization/evaluation/NIST_SRE2000/Disk8_ground_truth) | Multiple | $500.00 | [Evaluation Plan](https://www.nist.gov/system/files/documents/2017/09/26/spk-2000-plan-v1.0.htm_.pdf) |
| [2003 NIST Rich Transcription Evaluation Data](https://catalog.ldc.upenn.edu/LDC2007S10) | Together with audios | en, ar, zh | $2000.00 | Telephone speech, broadcast news. |
| [CALLHOME American English Speech](https://catalog.ldc.upenn.edu/LDC97S42) | [CALLHOME American English Transcripts](https://catalog.ldc.upenn.edu/LDC97T14) | en | $1500.00 + $1000.00 | [CH109 whitelist](https://github.com/google/speaker-id/blob/master/publications/LstmDiarization/evaluation/CALLHOME_American_English/ch109_whitelist.txt) |
| [The ICSI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/icsi/) | Together with audios | en | Free | ~72 hours (75 meetings). [License (CC BY 4.0)](https://groups.inf.ed.ac.uk/ami/icsi/license.shtml) |
| [The AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) | Together with audios (need to be processed; see [ami-diarization-setup](https://github.com/pyannote/ami-diarization-setup)) | en | Free | 100 hours of multi-accent English meetings. [License (CC BY 4.0)](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml) |
| [Fisher English Training Speech Part 1 Speech](https://catalog.ldc.upenn.edu/LDC2004S13) | [Fisher English Training Speech Part 1 Transcripts](https://catalog.ldc.upenn.edu/LDC2004T19) | en | $7000.00 + $1000.00 | Conversational telephone speech (5,850 calls, ~975 hours). |
| [Fisher English Training Part 2, Speech](https://catalog.ldc.upenn.edu/LDC2005S13) | [Fisher English Training Part 2, Transcripts](https://catalog.ldc.upenn.edu/LDC2005T19) | en | $7000.00 + $1000.00 | Conversational telephone speech (5,849 calls, ~975 hours). |
| [DIHARD III](https://catalog.ldc.upenn.edu/LDC2022S12) ([Eval](https://catalog.ldc.upenn.edu/LDC2022S14)) | Together with audios | en, zh | $300.00 + $300.00 | 68.15 hours (34.15h dev + 34.00h eval) across 11 diverse acoustic domains (clinical, courtroom, restaurant, web video, CTS, etc.). |
| [VoxConverse](https://github.com/joonson/voxconverse) | [Dev](https://github.com/joonson/voxconverse/tree/master/dev) & [Test](https://github.com/joonson/voxconverse/tree/master/test) RTTMs (v0.3) | Multiple (mostly en) | Free | Audio-visual diarisation dataset of 63.8 hours across 448 multispeaker YouTube clips (CC BY 4.0). |
| [CHiME-6](https://www.chimechallenge.org/) | Together with audios | en | Free | ~50 hours (20 4-speaker dinner-party sessions) recorded on distant Kinect microphone arrays and binaural microphones ([OpenSLR 150](https://www.openslr.org/150/)). |
| [AISHELL-4](https://www.openslr.org/111/) | Together with audios | zh | Free | 120 hours (211 sessions, 4–8 speakers per session) of real Mandarin multi-party meetings recorded on an 8-channel circular mic array + headsets (CC BY-SA 4.0). |
| [MiniVox Benchmark](https://github.com/doerlbh/MiniVox) | Generated automatically | en | Free | MiniVox is an automatic framework to transform any speaker-labelled dataset into continuous speech datastream with episodically revealed label feedbacks. |
| [The AliMeeting Corpus](https://www.openslr.org/119/) | Together with audios | zh | Free | 118.75 hours of real Mandarin meetings (far-field 8-ch mic array + near-field headsets, ~42% overlap in train; CC BY-SA 4.0). [Baseline repo](https://github.com/yufan-aslp/AliMeeting). |

### Speaker embedding training sets

| Name | Utterances | Speakers | Language | Pricing | Additional information |
| ---- | ---------- | -------- | -------- | ------- | ---------------------- |
| [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) | 6.3K | 630 | en | $250.00 | Published in 1993, the TIMIT corpus of read speech is one of the earliest speaker recognition datasets. |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | 44K | 110 | en | Free | CSTR VCTK Corpus (v0.92, CC BY 4.0). Newspaper sentences plus the Rainbow Passage and an elicitation paragraph across diverse English accents. |
| [LibriSpeech](https://www.openslr.org/12/) | 292K | 2.4K+ | en | Free | Large-scale (~1,000 hours) corpus of read English speech (CC BY 4.0). |
| [Multilingual LibriSpeech (MLS)](https://www.openslr.org/94/) | 12.3M | 6.3K+ | en, de, nl, es, fr, it, pt, pl | Free | Large-scale (~50.8K hours) multilingual corpus derived from read audiobooks from LibriVox across 8 languages - English, German, Dutch, Spanish, French, Italian, Portuguese, Polish (CC BY 4.0). |
| [LibriVox](https://librivox.org/) | 350K+ chapters | 14.9K+ | Multiple | Free | Free public-domain audiobooks (21.8K+ cataloged works across 49 languages). LibriSpeech and MLS are processed subsets of LibriVox. Each unsegmented chapter recording can be very long. |
| [VoxCeleb 1&2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 1.28M+ | 7.3K+ | Multiple | Free | Audio-visual dataset (~2,794 hours) consisting of short clips of human speech extracted from YouTube interview videos (CC BY-SA 4.0). |
| [The Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/) | 5.4K articles | 879 | en, de, nl | Free | Volunteer readers reading full Wikipedia articles (~1,005 hours total, CC BY-SA 4.0). |
| [CN-Celeb 1&2](https://www.openslr.org/82/) | 660K | 3K | zh | Free | Multi-genre Chinese speaker recognition corpus (~1,364 hours across 11 genres: CN-Celeb1 + CN-Celeb2; CC BY-SA 4.0) released by CSLT@Tsinghua University. |
| [BookTubeSpeech](https://users.wpi.edu/~jrwhitehill/BookTubeSpeech/index.html) | 8.4K videos | 8.4K | en | Free | Audio samples (~1,090 hours) extracted from BookTube YouTube videos (CC BY 4.0). The dataset can be downloaded using [BookTubeSpeech-download](https://github.com/wq2012/BookTubeSpeech-download). |
| [DeepMine](https://data.deepmine.ir/en/index.html) | 540K+ | 1,858 | fa, en | Paid (contact authors) | A speech database in Persian and English (>480 hours transcribed) designed to build and evaluate speaker verification and Persian ASR systems. |
| [NISP-Dataset](https://github.com/iiscleap/NISP-Dataset) | 28K+ | 345 | en, hi, kn, ml, ta, te | Free | Multi-accent Indian English and 5 Indian languages (56.9 hours, CC BY 4.0) along with speaker physical parameters (height, weight, ...) and regional/linguistic metadata. |
| [VoxBlink2](https://voxblink2.github.io/) | 9.9M | 111K+ | 18 languages (en, pt, es, ru, ar, ...) | Free | Multilingual dataset (~16,672 hours, CC BY-NC-SA 4.0) from [VoxBlink2: A 100K+ Speaker Recognition Corpus and the Open-Set Speaker-Identification Benchmark](https://arxiv.org/abs/2407.11510) |

### Augmentation noise sources

| Name | Utterances | Pricing | Additional information |
| ---- | ---------- | ------- | ---------------------- |
| [AudioSet](https://research.google.com/audioset/) | 2M+ | Free | A large-scale dataset of 2.08M human-annotated 10-second audio event clips (~5,789 hours; CC BY 4.0). |
| [MUSAN](https://www.openslr.org/17/) | 2K+ (~109 hours) | Free | Corpus of music (42.5h), speech (~60h), and noise (~6h, 929 files) recordings (CC BY 4.0). |
| [RIRS_NOISES](https://www.openslr.org/28/) | 60K+ RIRs & noises | Free | Simulated and real room impulse responses plus isotropic and point-source noises at 16 kHz (Apache 2.0). |

## Conferences

| Conference/Workshop | Frequency | Page Limit          | Organization     | Double-Blind Review |
| ------------------- | --------- | ------------------- | ---------------- | ------------------- |
| ICASSP              | Annual    | 4 + 1 (ref)         | IEEE             | No                  |
| Interspeech         | Annual    | 4 + 2 (ref)         | ISCA             | Yes                 |
| Speaker Odyssey     | Biennial  | 6 + 2 (ref)         | ISCA             | Yes                 |
| SLT                 | Biennial  | 6 + 2 (ref)         | IEEE             | Yes                 |
| ASRU                | Biennial  | 6 + 2 (ref)         | IEEE             | Yes                 |
| WASPAA              | Biennial  | 4 + 1 (ref)         | IEEE             | Yes                 |
| IJCB                | Annual    | 8 + unlimited (ref) | IEEE & IAPR TC-4 | Yes                 |

## Products

| Company | Product |
| ------- | ------- |
| Google | [Recorder app](https://support.google.com/pixelphone?p=recorder_speaker_labels) |
| Google | [Google Cloud Speech-to-Text API](https://docs.cloud.google.com/speech-to-text/docs/multiple-voices) |
| Google | [Gemini 3.5 Transcribe](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe) |
| Amazon | [Amazon Transcribe](https://aws.amazon.com/transcribe/) |
| Microsoft | [Azure AI Speech Diarization API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-stt-diarization) |
| IBM | [Watson Speech to Text](https://www.ibm.com/products/speech-to-text) |
| Alibaba | [Tongyi Tingwu (通义听悟)](https://tingwu.aliyun.com/) |
| pyannoteAI | [pyannoteAI](https://www.pyannote.ai/) - commercial cloud API and speaker diarization models from the creators of `pyannote.audio` |
| AssemblyAI | [Speaker Diarization API](https://www.assemblyai.com/docs/pre-recorded-audio/label-speakers) |
| Deepgram | [Diarization API](https://developers.deepgram.com/docs/diarization) |
| Speechmatics | [Speaker Diarization API](https://docs.speechmatics.com/speech-to-text/formatting) |
| ElevenLabs | [Scribe Speech to Text](https://elevenlabs.io/speech-to-text) |
| Kviskr | [Kviskr](https://kviskr.no) - macOS app for local dictation and meeting transcription with speaker diarization using NB-Whisper |
| diarize | [diarize.io](https://diarize.io) - hosted API and web app for speaker-labeled transcripts of YouTube and other video links |
| Screenpipe | [Screenpipe](https://screenpipe.com) - Desktop screen/audio history with speaker diarization, local storage, and optional cloud transcription (source-available, Screenpipe Commercial License) |

## Publications

### Special topics

#### Review and survey papers

* [A Review of Speaker Diarization: Recent Advances with Deep Learning](https://arxiv.org/abs/2101.09624), 2022
* [A review on speaker diarization systems and approaches](https://www.sciencedirect.com/science/article/abs/pii/S0167639312000696), 2012
* [Speaker diarization: A review of recent research](https://www.eurecom.fr/fr/publication/3152/download/mm-publi-3152.pdf), 2012

#### Large language model (LLM)

* [SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](https://arxiv.org/abs/2508.06372), 2026
* [DiarizationLM: Speaker Diarization Post-Processing with Large Language Models](https://arxiv.org/abs/2401.03506), 2024
* [Enhancing Speaker Diarization with Large Language Models: A Contextual Beam Search Approach](https://arxiv.org/abs/2309.05248), 2024
* [Lexical speaker error correction: Leveraging language models for speaker diarization error correction](https://arxiv.org/abs/2306.09313), 2023

#### Supervised diarization

* [DiaPer: End-to-End Neural Diarization with Perceiver-Based Attractors](https://arxiv.org/abs/2312.04324), 2024
* [Powerset multi-class cross entropy loss for neural speaker diarization](https://arxiv.org/abs/2310.13025), 2023
* [TOLD: A Novel Two-Stage Overlap-Aware Framework for Speaker Diarization](https://arxiv.org/abs/2303.05397), 2023
* [Encoder-Decoder Based Attractors for End-to-End Neural Diarization](https://arxiv.org/abs/2106.10654), 2022
* [Speaker Overlap-aware Neural Diarization for Multi-party Meeting Analysis](https://arxiv.org/abs/2211.10243), 2022
* [End-to-End Diarization for Variable Number of Speakers with Local-Global Networks and Discriminative Speaker Embeddings](https://arxiv.org/abs/2105.02096), 2021
* [Discriminative Neural Clustering for Speaker Diarisation](https://arxiv.org/abs/1910.09703), 2021
* [Supervised online diarization with sample mean loss for multi-domain data](https://arxiv.org/abs/1911.01266), 2020
* [End-to-End Neural Speaker Diarization with Permutation-Free Objectives](https://arxiv.org/abs/1909.05952), 2019
* [End-to-End Neural Speaker Diarization with Self-attention](https://arxiv.org/abs/1909.06247), 2019
* [Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719), 2019

#### Joint diarization and ASR

* [Train Short, Infer Long: Speech-LLM Enables Zero-Shot Streamable Joint ASR and Diarization on Long Audio](https://arxiv.org/abs/2511.16046), 2026
* [Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems](https://arxiv.org/abs/2409.06656), 2025
* [One Model to Rule Them All? Towards End-to-End Joint Speaker Diarization and Speech Recognition](https://arxiv.org/abs/2310.01688), 2024
* [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio](https://arxiv.org/abs/2303.00747), 2023
* [A Comparative Study on Speaker-attributed Automatic Speech Recognition in Multi-party Meetings](https://arxiv.org/abs/2203.16834), 2022
* [Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection](https://arxiv.org/abs/2109.11641), 2022
* [Transcribe-to-Diarize: Neural Speaker Diarization for Unlimited Number of Speakers using End-to-End Speaker-Attributed ASR](https://arxiv.org/abs/2110.03151), 2022
* [Joint Speech Recognition and Speaker Diarization via Sequence Transduction](https://arxiv.org/abs/1907.05337), 2019
* [Says who? Deep learning models for joint speech recognition, segmentation and diarization](https://ieeexplore.ieee.org/abstract/document/8462375), 2018

#### Online speaker diarization

* [Speaker Diarization as a Fully Online Bandit Learning Problem in MiniVox](https://proceedings.mlr.press/v157/lin21c/lin21c.pdf), 2021
* [VoiceID on the Fly: A Speaker Recognition System that Learns from Scratch](https://www.isca-archive.org/interspeech_2020/lin20b_interspeech.html), 2020

#### Challenges

* [The CHiME-8 DASR Challenge for Generalizable and Array Agnostic Distant Automatic Speech Recognition and Diarization](https://arxiv.org/abs/2407.16447), 2024
* [M2MeT: The ICASSP 2022 Multi-Channel Multi-Party Meeting Transcription Challenge](https://arxiv.org/abs/2110.07393), 2022
* [The Hitachi-JHU DIHARD III system: Competitive end-to-end neural diarization and x-vector clustering systems combined by DOVER-Lap](https://arxiv.org/abs/2102.01363), 2021
* [Diarization is Hard: Some Experiences and Lessons Learned for the JHU Team in the Inaugural DIHARD Challenge](https://www.isca-archive.org/interspeech_2018/sell18_interspeech.html), 2018
* [ODESSA at Albayzin Speaker Diarization Challenge 2018](https://www.isca-archive.org/iberspeech_2018/patino18_iberspeech.html), 2018
* [Joint Discriminative Embedding Learning, Speech Activity and Overlap Detection for the DIHARD Challenge](https://www.isca-archive.org/interspeech_2018/miasatofilho18_interspeech.html), 2018

#### Audio-Visual Speaker Diarization

* [3D-Speaker-Toolkit: An Open-Source Toolkit for Multimodal Speaker Verification and Diarization](https://arxiv.org/abs/2403.19971), 2025
* [AVA-AVD: Audio-Visual Speaker Diarization in the Wild](https://dl.acm.org/doi/abs/10.1145/3503161.3548027), 2022
* [DyViSE: Dynamic Vision-Guided Speaker Embedding for Audio-Visual Speaker Diarization](https://ieeexplore.ieee.org/document/9948860), 2022
* [End-to-End Audio-Visual Neural Speaker Diarization](https://www.isca-archive.org/interspeech_2022/he22c_interspeech.html), 2022
* [MSDWild: Multi-modal Speaker Diarization Dataset in the Wild](https://www.isca-archive.org/interspeech_2022/liu22t_interspeech.html), 2022


### Other

#### 2026

* [Efficient and Robust Speaker Diarization via Structured Pruning of Self-Supervised Models](https://arxiv.org/abs/2506.18623)

#### 2025

* [Leveraging Self-Supervised Learning for Speaker Diarization](https://arxiv.org/abs/2409.09408)

#### 2024

* [TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings](https://arxiv.org/abs/2303.03849)

#### 2023

* [pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe](https://www.isca-archive.org/interspeech_2023/bredin23_interspeech.html)

#### 2022

* [From Simulated Mixtures to Simulated Conversations as Training Data for End-to-End Neural Diarization](https://arxiv.org/abs/2204.00890)
* [Multi-scale Speaker Diarization with Dynamic Scale Weighting](https://arxiv.org/abs/2203.15974)

#### 2021

* [Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](https://arxiv.org/abs/2109.06483)
* [End-to-end speaker segmentation for overlap-aware resegmentation](https://www.isca-archive.org/interspeech_2021/bredin21_interspeech.html)
* [DIVE: End-to-end Speech Diarization via Iterative Speaker Embedding](https://arxiv.org/abs/2105.13802)
* [DOVER-Lap: A method for combining overlap-aware diarization outputs](https://arxiv.org/abs/2011.01997)
* [Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: Theory, implementation and analysis on standard tasks](https://www.sciencedirect.com/science/article/pii/S0885230821000619)
* [AISHELL-4: An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario](https://arxiv.org/abs/2104.03603)

#### 2020

* [Spot the conversation: speaker diarisation in the wild](https://arxiv.org/abs/2007.01216)
* [Speaker Diarization with Region Proposal Network](https://arxiv.org/abs/2002.06220)
* [Target-Speaker Voice Activity Detection: a Novel Approach for Multi-Speaker Diarization in a Dinner Party Scenario](https://arxiv.org/abs/2005.07272)
* [Overlap-aware diarization: resegmentation using neural end-to-end overlapped speech detection](https://arxiv.org/abs/1910.11646)
* [Speaker diarization using latent space clustering in generative adversarial network](https://arxiv.org/abs/1910.11398)

#### 2019

* [A study of semi-supervised speaker diarization system using gan mixture model](https://arxiv.org/abs/1910.11416)
* [Enhancements for Audio-only Diarization Systems](https://arxiv.org/abs/1909.00082)
* [LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization](https://arxiv.org/abs/1907.10393)
* [Meeting Transcription Using Virtual Microphone Arrays](https://arxiv.org/abs/1905.02545)
* [Speaker diarisation using 2D self-attentive combination of embeddings](https://arxiv.org/abs/1902.03190)
* [Speaker Diarization with Lexical Information](https://arxiv.org/abs/2004.06756)

#### 2018

* [Speaker Diarization with LSTM](https://arxiv.org/abs/1710.10468)
* [Neural speech turn segmentation and affinity propagation for speaker diarization](https://www.isca-archive.org/interspeech_2018/yin18b_interspeech.html)
* [Multimodal Speaker Segmentation and Diarization using Lexical and Acoustic Cues via Sequence to Sequence Neural Networks](https://arxiv.org/abs/1805.10731)

#### 2017

* [Speaker diarization using deep neural network embeddings](https://www.danielpovey.com/files/2017_icassp_diarization_embeddings.pdf)
* [Speaker diarization using convolutional neural network for statistics accumulation refinement](https://www.isca-archive.org/interspeech_2017/zajic17_interspeech.html)
* [pyannote.metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems](https://www.isca-archive.org/interspeech_2017/bredin17_interspeech.html)
* [Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks](https://www.isca-archive.org/interspeech_2017/yin17_interspeech.html)
* [Speaker Diarization using Deep Recurrent Convolutional Neural Networks for Speaker Embeddings](https://arxiv.org/abs/1708.02840)

#### 2016

* [A Speaker Diarization System for Studying Peer-Led Team Learning Groups](https://arxiv.org/abs/1606.07136)

#### 2015

* [Diarization resegmentation in the factor analysis subspace](https://ieeexplore.ieee.org/document/7178881)

#### 2014

* [A study of the cosine distance-based mean shift for telephone speech diarization](https://ieeexplore.ieee.org/document/6633085)
* [Speaker diarization with PLDA i-vector scoring and unsupervised calibration](https://ieeexplore.ieee.org/abstract/document/7078610)
* [Artificial neural network features for speaker diarization](https://ieeexplore.ieee.org/abstract/document/7078608)

#### 2013

* [Unsupervised methods for speaker diarization: An integrated and iterative approach](https://sls.csail.mit.edu/publications/2013/Shum_IEEE_Oct-2013.pdf)

#### 2011

* [PLDA-based Clustering for Speaker Diarization of Broadcast Streams](https://www.isca-archive.org/interspeech_2011/silovsky11_interspeech.html)
* [Speaker diarization of meetings based on speaker role n-gram models](https://publications.idiap.ch/downloads/papers/2011/Valente_ICASSP2011_2011.pdf)

#### 2009

* [Speaker Diarization for Meeting Room Audio](https://www.isca-archive.org/interspeech_2009/sun09b_interspeech.html)

#### 2008

* [Stream-based speaker segmentation using speaker factors and eigenvoices](https://ieeexplore.ieee.org/document/4518564)

#### 2006

* [An overview of automatic speaker diarization systems](https://ieeexplore.ieee.org/document/1677976)
* [A spectral clustering approach to speaker diarization](https://www.isca-archive.org/interspeech_2006/ning06_interspeech.html)

## Other learning materials

### Online courses

* Course on Udemy: [A Tutorial on Speaker Diarization](https://www.udemy.com/course/diarization/?referralCode=21D7CC0AEABB7FE3680F)

### Books

* [Voice Identity Techniques: From core algorithms to engineering practice (Chinese)](https://github.com/wq2012/VoiceIdentityBook) by Quan Wang, 2020

### Tech blogs

* [Literature Review For Speaker Change Detection](https://halil-erdogan.github.io/hedonistrh.github.io/2018-07-09-Literature-Review-for-Speaker-Change-Detection/)
  by [Halil Erdoğan](https://github.com/halil-erdogan)
* [Speaker Diarization: Separation of Multiple Speakers in an Audio File](https://medium.com/datadriveninvestor/speaker-diarization-22121f1264b1) by [Jaspreet Singh](https://medium.com/@jaspreetuseducation)
* [Speaker Diarization with Kaldi](https://medium.com/towards-data-science/speaker-diarization-with-kaldi-e30301b05cc8) by [Yoav Ramon](https://medium.com/@yoavramon)
* [Who spoke when! How to Build your own Speaker Diarization Module](https://medium.com/saarthi-ai/who-spoke-when-build-your-own-speaker-diarization-module-from-scratch-e7d725ee279) by Rahul Saxena

### Video tutorials

* [pyannote audio: neural building blocks for speaker diarization](https://www.youtube.com/watch?v=37R_R82lfwA) by Hervé Bredin
* [Google's Diarization System: Speaker Diarization with LSTM](https://www.youtube.com/watch?v=pjxGPZQeeO4) by Google
* [Fully Supervised Speaker Diarization: Say Goodbye to clustering](https://www.youtube.com/watch?v=pGkqwRPzx9U) by Google
* [Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection](https://www.youtube.com/watch?v=U79Aw1ky7ag) by Google
* [Speaker Diarization: Optimal Clustering and Learning Speaker Embeddings](https://www.youtube.com/watch?v=vcyB8xb1-ys) by Microsoft Research
* [Robust Speaker Diarization for Meetings: the ICSI system](https://www.youtube.com/watch?v=kEcUcfLmIS0) by Microsoft Research
* [【机器之心&博文视点】入门声纹技术｜第二讲：声纹分割聚类与其他应用](https://www.youtube.com/watch?v=HE9JW8yKYRk) by Quan Wang

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=wq2012/awesome-diarization&type=Date)](https://www.star-history.com/#wq2012/awesome-diarization&Date)
