# Awesome Speaker Diarization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome) [![Contribution](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/wq2012/awesome-diarization/blob/master/CONTRIBUTING.md)

## Table of contents

* [Overview](#Overview)
* [Publications](#Publications)
* [Software](#Software)
  * [Framework](#Framework)
  * [Evaluation](#Evaluation)
  * [Clustering](#Clustering)
  * [Speaker embedding](#Speaker-embedding)
  * [Speaker change detection](#Speaker-change-detection)
  * [Other sotware](#Other-software)
* [Datasets](#Datasets)
  * [Diarization datasets](#Diarization-datasets)
  * [Speaker embedding training sets](#Speaker-embedding-training-sets)
* [Leaderboards](#Leaderboards)
* [Other learning materials](#Other-learning-materials)
  * [Tech blogs](#Tech-blogs)
  * [Video tutorials](#Video-tutorials)
* [Products](#Products)


## Overview

This is a curated list of awesome Speaker Diarization papers, libraries, datasets, and other resources.

The purpose of this repo is to organize the world’s resources for speaker diarization, and make them universally accessible and useful.

To add items to this page, simply send a pull request. ([contributing guide](CONTRIBUTING.md))

## Publications

### Special topics

#### Supervisied diarization

* [Supervised online diarization with sample mean loss for multi-domain data](https://arxiv.org/abs/1911.01266), 2019
* [Discriminative Neural Clustering for Speaker Diarisation](https://arxiv.org/abs/1910.09703), 2019
* [End-to-End Neural Speaker Diarization with Permutation-Free Objectives](https://arxiv.org/abs/1909.05952), 2019
* [End-to-End Neural Speaker Diarization with Self-attention](https://arxiv.org/abs/1909.06247), 2019
* [Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719), 2018

#### Joint diarization and ASR

* [Joint Speech Recognition and Speaker Diarization via Sequence Transduction](https://arxiv.org/abs/1907.05337), 2019
* [Says who? Deep learning models for joint speech recognition, segmentation and diarization](https://ieeexplore.ieee.org/abstract/document/8462375), 2018

#### Challenges

* [Diarization is Hard: Some Experiences and Lessons Learned for the JHU
Team in the Inaugural DIHARD Challenge](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1893.pdf), 2018
* [ODESSA at Albayzin Speaker Diarization Challenge 2018](https://www.isca-speech.org/archive/IberSPEECH_2018/pdfs/IberS18_AE-5_Patino.pdf), 2018
* [Joint Discriminative Embedding Learning, Speech Activity and Overlap Detection for the DIHARD Challenge](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2304.pdf), 2018

### Other

#### 2020

* [Speaker Diarization with Region Proposal Network](https://arxiv.org/abs/2002.06220)

#### 2019

* [Overlap-aware diarization: resegmentation using neural end-to-end overlapped speech detection](https://arxiv.org/abs/1910.11646)
* [Speaker diarization using latent space clustering in generative adversarial network](https://arxiv.org/abs/1910.11398)
* [A study of semi-supervised speaker diarization system using gan mixture model](https://arxiv.org/abs/1910.11416)
* [Learning deep representations by multilayer bootstrap networks for speaker diarization](https://arxiv.org/abs/1910.10969)
* [Enhancements for Audio-only Diarization Systems](https://arxiv.org/abs/1909.00082)
* [LSTM based Similarity Measurement with Spectral Clustering for Speaker Diarization](https://arxiv.org/abs/1907.10393)
* [Meeting Transcription Using Virtual Microphone Arrays](https://www.microsoft.com/en-us/research/uploads/prod/2019/05/DenmarkTechReport-5ccb8b095c8f3.pdf)
* [Speaker diarisation using 2D self-attentive combination of embeddings](https://arxiv.org/abs/1902.03190)

#### 2018

* [Neural speech turn segmentation and affinity propagation for speaker diarization](https://hal.archives-ouvertes.fr/hal-01912236/)
* [Multimodal Speaker Segmentation and Diarization using Lexical and Acoustic Cues via Sequence to Sequence Neural Networks](https://arxiv.org/abs/1805.10731)
* [Joint Speaker Diarization and Recognition Using Convolutional and Recurrent Neural Networks](https://ieeexplore.ieee.org/abstract/document/8461666)

#### 2017

* [Speaker Diarization with LSTM](https://arxiv.org/abs/1710.10468)
* [Speaker diarization using deep neural network embeddings](http://danielpovey.com/files/2017_icassp_diarization_embeddings.pdf)
* [Speaker diarization using convolutional neural network for statistics accumulation refinement](https://pdfs.semanticscholar.org/35c4/0fde977932d8a3cd24f5a1724c9dbca8b38d.pdf)
* [pyannote. metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0411.PDF)
* [Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks](https://pdfs.semanticscholar.org/edff/b62b32ffcc2b5cc846e26375cb300fac9ecc.pdf)
* [Speaker Diarization using Deep Recurrent Convolutional Neural Networks for Speaker Embeddings](https://arxiv.org/abs/1708.02840)

#### 2016

* [A Speaker Diarization System for Studying Peer-Led Team Learning Groups](https://arxiv.org/pdf/1606.07136.pdf)

#### 2015

* [Diarization resegmentation in the factor analysis subspace](https://engineering.jhu.edu/hltcoe/wp-content/uploads/sites/92/2016/10/Sell_Garcia-Romero_2015A.pdf)

#### 2014

* [A study of the cosine distance-based mean shift for telephone speech diarization](https://www.researchgate.net/profile/Patrick_Kenny/publication/260661427_A_Study_of_the_Cosine_Distance-Based_Mean_Shift_for_Telephone_Speech_Diarization/links/0c96053270d2eaa133000000.pdf)
* [Speaker diarization with PLDA i-vector scoring and unsupervised calibration](https://ieeexplore.ieee.org/abstract/document/7078610)
* [Artificial neural network features for speaker diarization](https://ieeexplore.ieee.org/abstract/document/7078608)

#### 2013
* [Unsupervised methods for speaker diarization: An integrated and iterative approach](http://groups.csail.mit.edu/sls/publications/2013/Shum_IEEE_Oct-2013.pdf)

#### 2011

* [PLDA-based Clustering for Speaker Diarization of Broadcast Streams](https://pdfs.semanticscholar.org/0175/a752c5c72cadc7c0b899fd15f2f6b93c3335.pdf)
* [Speaker diarization of meetings based on speaker role n-gram models](https://publications.idiap.ch/downloads/papers/2011/Valente_ICASSP2011_2011.pdf)

#### 2010

* [Speaker diarization: A review of recent research](http://www.eurecom.fr/fr/publication/3152/download/mm-publi-3152.pdf)

#### 2009

* [Speaker Diarization for Meeting Room Audio](https://wiki.inf.ed.ac.uk/twiki/pub/CSTR/ListenSemester2_2009_10/sun_IS2009_SpDia_meeting.PDF)

#### 2008

* [Stream-based speaker segmentation using speaker factors and eigenvoices](https://www.researchgate.net/profile/Pietro_Laface/publication/224313019_Stream-based_speaker_segmentation_using_speaker_factors_and_eigenvoices/links/5770fe8608ae10de639dc121.pdf)

#### 2006

* [An overview of automatic speaker diarization systems](https://alize.univ-avignon.fr/doc/publis/06_IEEE-TASP_Tranter.pdf)
* [A spectral clustering approach to speaker diarization](http://www.ifp.illinois.edu/~hning2/papers/Ning_spectral.pdf)

## Software

### Framework

| Link | Language | Description |
| ---- | -------- | ----------- |
| [SIDEKIT for diarization (s4d)](https://projets-lium.univ-lemans.fr/s4d/) | Python | An open source package extension of SIDEKIT for Speaker diarization. |
| [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) ![GitHub stars](https://img.shields.io/github/stars/tyiannak/pyAudioAnalysis?style=social) | Python | Python Audio Analysis Library: Feature Extraction, Classification, Segmentation and Applications. |
| [AaltoASR](https://github.com/aalto-speech/speaker-diarization) ![GitHub stars](https://img.shields.io/github/stars/aalto-speech/speaker-diarization?style=social) | Python & Perl | Speaker diarization scripts, based on AaltoASR. |
| [LIUM SpkDiarization](https://projets-lium.univ-lemans.fr/spkdiarization/) | Java | LIUM_SpkDiarization is a software dedicated to speaker diarization (i.e. speaker segmentation and clustering). It is written in Java, and includes the most recent developments in the domain (as of 2013). |
| [kaldi-asr](https://github.com/kaldi-asr/kaldi/tree/master/egs/callhome_diarization) [![Build Status](https://travis-ci.com/kaldi-asr/kaldi.svg?branch=master)](https://travis-ci.com/kaldi-asr/kaldi) | Bash | Example scripts for speaker diarization on a portion of CALLHOME used in the 2000 NIST speaker recognition evaluation. |
| [Alize LIA_SpkSeg](https://alize.univ-avignon.fr/) | C++ | ALIZÉ is an opensource platform for speaker recognition. LIA_SpkSeg is the tools for speaker diarization. |
| [pyannote-audio](https://github.com/pyannote/pyannote-audio) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-audio?style=social) | Python | Neural building blocks for speaker diarization: speech activity detection, speaker change detection, speaker embedding. |
| [pyBK](https://github.com/josepatino/pyBK) ![GitHub stars](https://img.shields.io/github/stars/josepatino/pyBK?style=social) | Python | Speaker diarization using binary key speaker modelling. Computationally light solution that does not require external training data. |
| [Speaker-Diarization](https://github.com/taylorlu/Speaker-Diarization) ![GitHub stars](https://img.shields.io/github/stars/taylorlu/Speaker-Diarization?style=social) | Python | Speaker diarization using uis-rnn and GhostVLAD. An easier way to support openset speakers. |
| [EEND](https://github.com/hitachi-speech/EEND) ![GitHub stars](https://img.shields.io/github/stars/hitachi-speech/EEND?style=social) | Python & Bash & Perl | End-to-End Neural Diarization. |
| [VBDiarization](https://github.com/Jamiroquai88/VBDiarization) ![GitHub stars](https://img.shields.io/github/stars/Jamiroquai88/VBDiarization?style=social) | Python | Speaker diarization based on Kaldi x-vectors using pretrained model trained in Kaldi ([kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi)) and converted to ONNX format ([onnx/onnx](https://github.com/onnx/onnx)) running in ONNXRuntime ([Microsoft/onnxruntime](https://github.com/Microsoft/onnxruntime)). |
| [RE-VERB](https://github.com/team-re-verb/RE-VERB) ![GitHub stars](https://img.shields.io/github/stars/team-re-verb/RE-VERB?style=social) | Python & JavaScript | RE: VERB is speaker diarization system, it allows the user to send/record audio of a conversation and receive timestamps of who spoke when. |

### Evaluation

| Link | Language | Description |
| ---- | -------- | ----------- |
| [pyannote-metrics](https://github.com/pyannote/pyannote-metrics) ![GitHub stars](https://img.shields.io/github/stars/pyannote/pyannote-metrics?style=social) [![Build Status](https://travis-ci.org/pyannote/pyannote-metrics.svg?branch=master)](https://travis-ci.org/pyannote/pyannote-metrics)  | Python| A toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems. |
| [SimpleDER](https://github.com/wq2012/SimpleDER) ![GitHub stars](https://img.shields.io/github/stars/wq2012/SimpleDER?style=social) [![Build Status](https://travis-ci.org/wq2012/SimpleDER.svg?branch=master)](https://travis-ci.org/wq2012/SimpleDER) | Python | A lightweight library to compute Diarization Error Rate (DER). |
| NIST md-eval | Perl | (1) modified [md-eval.pl](http://www1.icsi.berkeley.edu/~knoxm/dia/) from [Mary Tai Knox](http://www1.icsi.berkeley.edu/~knoxm); (2) [md-eval-v21.pl](https://github.com/jitendrab/btp/blob/master/c_code/single_diag_gaussian_no_viterbi/md-eval-v21.pl) from [jitendra](https://github.com/jitendrab); (3) [md-eval-22.pl](https://github.com/nryant/dscore/blob/master/scorelib/md-eval-22.pl) from [nryant](https://github.com/nryant) |
| [dscore](https://github.com/nryant/dscore) ![GitHub stars](https://img.shields.io/github/stars/nryant/dscore?style=social) | Python & Perl | Diarization scoring tools. |
| [Sequence Match Accuracy](https://github.com/google/uis-rnn/blob/master/uisrnn/evals.py) | Python | Match the accuracy of two sequences with Hungarian algorithm. |

### Clustering

| Link | Language | Description |
| ---- | -------- | ----------- |
| [uis-rnn](https://github.com/google/uis-rnn) ![GitHub stars](https://img.shields.io/github/stars/google/uis-rnn?style=social) [![Build Status](https://travis-ci.org/google/uis-rnn.svg?branch=master)](https://travis-ci.org/google/uis-rnn) | Python & PyTorch | Google's Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN) algorithm, for Fully Supervised Speaker Diarization. This clustering algorithm is **supervised**. |
| [uis-rnn-sml](https://github.com/DonkeyShot21/uis-rnn-sml) ![GitHub stars](https://img.shields.io/github/stars/DonkeyShot21/uis-rnn-sml?style=social) | Python & PyTorch | A variant of UIS-RNN, for the paper Supervised Online Diarization with Sample Mean Loss for Multi-Domain Data. |
| [DNC](https://github.com/FlorianKrey/DNC) ![GitHub stars](https://img.shields.io/github/stars/FlorianKrey/DNC?style=social) | Python & ESPnet | Transformer-based Discriminative Neural Clustering (DNC) for Speaker Diarisation. Like UIS-RNN, it is **supervised**. |
| [SpectralCluster](https://github.com/wq2012/SpectralCluster) ![GitHub stars](https://img.shields.io/github/stars/wq2012/SpectralCluster?style=social) [![Build Status](https://travis-ci.org/wq2012/SpectralCluster.svg?branch=master)](https://travis-ci.org/wq2012/SpectralCluster) | Python | Spectral clustering with affinity matrix refinement operations. |
| [sklearn.cluster](https://scikit-learn.org/stable/modules/clustering.html) [![Build Status]( https://api.travis-ci.org/scikit-learn/scikit-learn.svg?branch=master)](https://travis-ci.org/scikit-learn/scikit-learn) | Python | scikit-learn clustering algorithms. |
| [PLDA](https://github.com/RaviSoji/plda) ![GitHub stars](https://img.shields.io/github/stars/RaviSoji/plda?style=social) | Python | Probabilistic Linear Discriminant Analysis & classification, written in Python. |
| [PLDA](https://github.com/mrouvier/plda) ![GitHub stars](https://img.shields.io/github/stars/mrouvier/plda?style=social) | C++ | Open-source implementation of simplified PLDA (Probabilistic Linear Discriminant Analysis). |

### Speaker embedding

| Link | Method | Language | Description |
| ---- | ------ | -------- | ----------- |
| [resemble-ai/Resemblyzer](https://github.com/resemble-ai/Resemblyzer) ![GitHub stars](https://img.shields.io/github/stars/resemble-ai/Resemblyzer?style=social) | d-vector | Python & PyTorch | PyTorch implementation of generalized end-to-end loss for speaker verification, which can be used for voice cloning and diarization. |
| [Speaker_Verification](https://github.com/Janghyun1230/Speaker_Verification) ![GitHub stars](https://img.shields.io/github/stars/Janghyun1230/Speaker_Verification?style=social) | d-vector | Python & TensorFlow | Tensorflow implementation of generalized end-to-end loss for speaker verification. |
| [PyTorch_Speaker_Verification](https://github.com/HarryVolek/PyTorch_Speaker_Verification) ![GitHub stars](https://img.shields.io/github/stars/HarryVolek/PyTorch_Speaker_Verification?style=social) | d-vector | Python & PyTorch | PyTorch implementation of "Generalized End-to-End Loss for Speaker Verification" by Wan, Li et al. With UIS-RNN integration. |
| [Real-Time Voice Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) ![GitHub stars](https://img.shields.io/github/stars/CorentinJ/Real-Time-Voice-Cloning?style=social) | d-vector | Python & PyTorch | Implementation of "Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis" (SV2TTS) with a vocoder that works in real-time. |
| [deep-speaker](https://github.com/philipperemy/deep-speaker) ![GitHub stars](https://img.shields.io/github/stars/philipperemy/deep-speaker?style=social) | d-vector |Python & Keras | Third party implementation of the Baidu paper Deep Speaker: an End-to-End Neural Speaker Embedding System. |
| [x-vector-kaldi-tf](https://github.com/hsn-zeinali/x-vector-kaldi-tf) ![GitHub stars](https://img.shields.io/github/stars/hsn-zeinali/x-vector-kaldi-tf?style=social) | x-vector | Python & TensorFlow & Perl | Tensorflow implementation of x-vector topology on top of Kaldi recipe. |
| [kaldi-ivector](https://github.com/idiap/kaldi-ivector) ![GitHub stars](https://img.shields.io/github/stars/idiap/kaldi-ivector?style=social) | i-vector | C++ & Perl |  Extension to Kaldi implementing the standard i-vector hyperparameter estimation and i-vector extraction procedure. |
| [voxceleb-ivector](https://github.com/swshon/voxceleb-ivector) ![GitHub stars](https://img.shields.io/github/stars/swshon/voxceleb-ivector?style=social) | i-vector |Perl | Voxceleb1 i-vector based speaker recognition system. |

### Speaker change detection

| Link  | Language | Description |
| ----  | -------- | ----------- |
| [change_detection](https://github.com/yinruiqing/change_detection) ![GitHub stars](https://img.shields.io/github/stars/yinruiqing/change_detection?style=social) | Python & Keras | Code for Speaker Change Detection in Broadcast TV using Bidirectional Long Short-Term Memory Networks. |

### Other software

| Link | Language | Description |
| ---- | -------- | ----------- |
| [VB Diarization](https://github.com/wq2012/VB_diarization) ![GitHub stars](https://img.shields.io/github/stars/wq2012/VB_diarization?style=social) [![Build Status](https://travis-ci.org/wq2012/VB_diarization.svg?branch=master)](https://travis-ci.org/wq2012/VB_diarization) | Python | VB Diarization with Eigenvoice and HMM Priors. |

## Datasets

### Diarization datasets

| Audio | Diarization ground truth | Language | Pricing | Additional information |
| ----- | ------------------------ | -------- | ------- | ---------------------- |
| [2000 NIST Speaker Recognition Evaluation](https://catalog.ldc.upenn.edu/LDC2001S97) | [Disk-6 (Switchboard)](https://github.com/google/speaker-id/tree/master/publications/LstmDiarization/evaluation/NIST_SRE2000/Disk6_ground_truth), [Disk-8  (CALLHOME)](https://github.com/google/speaker-id/tree/master/publications/LstmDiarization/evaluation/NIST_SRE2000/Disk8_ground_truth) | Multiple | $2400.00 | [Evaluation Plan](https://www.nist.gov/sites/default/files/documents/2017/09/26/spk-2000-plan-v1.0.htm_.pdf) |
| [2003 NIST Rich Transcription Evaluation Data](https://catalog.ldc.upenn.edu/LDC2007S10) | Together with audios | en, ar, zh | $2000.00 | telephone speech, broadcast news |
| [CALLHOME American English Speech](https://catalog.ldc.upenn.edu/LDC97S42) | [CALLHOME American English Transcripts](https://catalog.ldc.upenn.edu/LDC97T14) | en | $1500.00 + $1000.00| [CH109 whitelist](https://github.com/google/speaker-id/blob/master/publications/LstmDiarization/evaluation/CALLHOME_American_English/ch109_whitelist.txt) |
| [The ICSI Meeting Corpus](http://groups.inf.ed.ac.uk/ami/icsi/) | Together with audios | en | Free | [License](http://groups.inf.ed.ac.uk/ami/icsi/license.shtml) |
| [The AMI Meeting Corpus](http://groups.inf.ed.ac.uk/ami/corpus/) | Together with audios (need to be processed) | Multiple | Free | [License](http://groups.inf.ed.ac.uk/ami/corpus/license.shtml) |
| [Fisher English Training Speech Part 1 Speech](https://catalog.ldc.upenn.edu/LDC2004S13) | [Fisher English Training Speech Part 1 Transcripts](https://catalog.ldc.upenn.edu/LDC2004T19)| en | $7000.00 + $1000.00 |
| [Fisher English Training Part 2, Speech](https://catalog.ldc.upenn.edu/LDC2005S13) | [Fisher English Training Part 2, Transcripts](https://catalog.ldc.upenn.edu/LDC2005T19) | en | $7000.00 + $1000.00 |

### Speaker embedding training sets

| Name | Utterances | Speakers | Language | Pricing | Additional information |
| ---- | ---------- | -------- | -------- | ------- | ---------------------- |
| [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1) | 6K+ | 630 | en | $250.00 | Published in 1993, the TIMIT corpus of read speech is one of the earliest speaker recognition datasets. |
| [VCTK](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html) | 43K+ | 109 | en | Free | Most were selected from a newspaper plus the Rainbow Passage and an elicitation paragraph intended to identify the speaker's accent. |
| [LibriSpeech](http://www.openslr.org/12) | 292K | 2K+ | en | Free | Large-scale (1000 hours) corpus of read English speech. |
| [LibriVox](https://librivox.org/) | 180K | 9K+ | Multiple | Free | Free public domain audiobooks. LibriSpeech is a processed subset of LibriVox. Each original unsegmented utterance could be very long. |
| [VoxCeleb 1&2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 1M+ | 7K | Multiple | Free | VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. |
| [The Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/) | 5K | 879 | en, de, nl | Free | Volunteer readers reading Wikipedia articles. |
| [CN-Celeb](http://www.openslr.org/82/) | 130K+ | 1K | zh | Free | A Free Chinese Speaker Recognition Corpus Released by CSLT@Tsinghua University. |
| [BookTubeSpeech](https://users.wpi.edu/~jrwhitehill/BookTubeSpeech/index.html) | 8K | 8K | en | Free | Audio samples extracted from BookTube videos - videos where people share their opinions on books - from YouTube. The dataset can be downloaded using [BookTubeSpeech-download](https://github.com/wq2012/BookTubeSpeech-download). |
| [DeepMine](http://data.deepmine.ir/en/index.html) | 540K | 1850 | fa, en | Unknown | A speech database in Persian and English designed to build and evaluate speaker verification, as well as Persian ASR systems. |

## Leaderboards

* [StateOfTheArt.ai](https://www.stateoftheart.ai/?area=Sound&task=Speaker%20Diarization)
* [DIHARD 2018](https://coml.lscp.ens.fr/dihard/2018/results.php)

## Other learning materials

### Tech blogs

* [Literature Review For Speaker Change Detection](https://hedonistrh.github.io/2018-07-09-Literature-Review-for-Speaker-Change-Detection/)
  by [Halil Erdoğan](https://github.com/hedonistrh)
* [Speaker Diarization: Separation of Multiple Speakers in an Audio File](https://medium.com/datadriveninvestor/speaker-diarization-22121f1264b1) by [Jaspreet Singh](https://medium.com/@jaspreetuseducation)
* [Speaker Diarization with Kaldi](https://towardsdatascience.com/speaker-diarization-with-kaldi-e30301b05cc8) by [Yoav Ramon](https://towardsdatascience.com/@yoavramon)

### Video tutorials

* [Google's Diarization System: Speaker Diarization with LSTM](https://www.youtube.com/watch?v=pjxGPZQeeO4) by Google
* [Fully Supervised Speaker Diarization: Say Goodbye to clustering](https://www.youtube.com/watch?v=pGkqwRPzx9U) by Google
* [Speaker Diarization: Optimal Clustering and Learning Speaker Embeddings](https://www.youtube.com/watch?v=vcyB8xb1-ys) by Microsoft Research
* [Robust Speaker Diarization for Meetings: the ICSI system](https://www.youtube.com/watch?v=kEcUcfLmIS0) by Microsoft Research

## Products

| Company | Product |
| ------- | ------- |
| Google  | [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text/docs/multiple-voices) |
| Amazon  | [Amazon Transcribe](https://aws.amazon.com/transcribe) |
| IBM     | [Watson Speech To Text API](https://www.ibm.com/watson/services/speech-to-text) |
| DeepAffects | [Speaker Diarization API](https://www.deepaffects.com/diarization-api) |
