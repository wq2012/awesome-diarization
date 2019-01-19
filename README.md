# Awesome Speaker Diarization [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

## Publications

### 2018

* [Fully Supervised Speaker Diarization](https://arxiv.org/abs/1810.04719)

### 2017

* [Speaker Diarization with LSTM](https://arxiv.org/abs/1710.10468)
* [Speaker diarization using deep neural network embeddings](http://danielpovey.com/files/2017_icassp_diarization_embeddings.pdf)
* [Speaker diarization using convolutional neural network for statistics accumulation refinement](https://pdfs.semanticscholar.org/35c4/0fde977932d8a3cd24f5a1724c9dbca8b38d.pdf)
* [pyannote. metrics: a toolkit for reproducible evaluation, diagnostic, and error analysis of speaker diarization systems](https://www.isca-speech.org/archive/Interspeech_2017/pdfs/0411.PDF)

### 2015

* [Diarization resegmentation in the factor analysis subspace](https://engineering.jhu.edu/hltcoe/wp-content/uploads/sites/92/2016/10/Sell_Garcia-Romero_2015A.pdf)

### 2014

* [A study of the cosine distance-based mean shift for telephone speech diarization](https://www.researchgate.net/profile/Patrick_Kenny/publication/260661427_A_Study_of_the_Cosine_Distance-Based_Mean_Shift_for_Telephone_Speech_Diarization/links/0c96053270d2eaa133000000.pdf)
* [Speaker diarization with PLDA i-vector scoring and unsupervised calibration](https://ieeexplore.ieee.org/abstract/document/7078610)
* [Artificial neural network features for speaker diarization](https://ieeexplore.ieee.org/abstract/document/7078608)

### 2013
* [Unsupervised methods for speaker diarization: An integrated and iterative approach](http://groups.csail.mit.edu/sls/publications/2013/Shum_IEEE_Oct-2013.pdf)

### 2011

* [PLDA-based Clustering for Speaker Diarization of Broadcast Streams](https://pdfs.semanticscholar.org/0175/a752c5c72cadc7c0b899fd15f2f6b93c3335.pdf)

### 2008

* [Stream-based speaker segmentation using speaker factors and eigenvoices](https://www.researchgate.net/profile/Pietro_Laface/publication/224313019_Stream-based_speaker_segmentation_using_speaker_factors_and_eigenvoices/links/5770fe8608ae10de639dc121.pdf)

## Source code

### Framework

* [SIDEKIT for diarization (s4d)](https://projets-lium.univ-lemans.fr/s4d/):
  An open source package extension of SIDEKIT for Speaker diarization.
* [LIUM_SpkDiarization](http://www-lium.univ-lemans.fr/diarization/doku.php/overview):
  LIUM_SpkDiarization is a software dedicated to speaker diarization
  (i.e. speaker segmentation and clustering). It is written in Java,
  and includes the most recent developments in the domain (as of 2013).

### Evaluation

* [pyannote-metrics](https://github.com/pyannote/pyannote-metrics): A toolkit
  for reproducible evaluation, diagnostic, and error analysis of speaker
  diarization systems.
* NIST `md-eval.pl` script:
  * [modified md-eval.pl](http://www1.icsi.berkeley.edu/~knoxm/dia/) by
    [Mary Tai Knox](http://www1.icsi.berkeley.edu/~knoxm)
  * [md-eval-v21.pl](https://github.com/jitendrab/btp/blob/master/c_code/single_diag_gaussian_no_viterbi/md-eval-v21.pl)
    from [jitendra](https://github.com/jitendrab)
* [dscore](https://github.com/nryant/dscore): Diarization scoring tools.


### Clustering

* [uis-rnn](https://github.com/google/uis-rnn): Google's
  Unbounded Interleaved-State Recurrent Neural Network (UIS-RNN) algorithm,
  for Fully Supervised Speaker Diarization
* [SpectralCluster](https://github.com/wq2012/SpectralCluster): Spectral
  clustering with affinity matrix refinement operations.

### Speaker embedding

* [d-vector in TensorFlow](https://github.com/Janghyun1230/Speaker_Verification)
* [d-vector in PyTorch](https://github.com/HarryVolek/PyTorch_Speaker_Verification)
* [x-vector in TensorFlow and Kaldi](https://github.com/hsn-zeinali/x-vector-kaldi-tf)
* [kaldi-ivector](https://github.com/idiap/kaldi-ivector)
* [voxceleb-ivector](https://github.com/swshon/voxceleb-ivector)

## Other learning materials

### Tech blog

* [Literature Review For Speaker Change Detection](https://hedonistrh.github.io/2018-07-09-Literature-Review-for-Speaker-Change-Detection/)
  by [Halil Erdoğan](https://github.com/hedonistrh)

### Video tutorials

* [Google's Diarization System: Speaker Diarization with LSTM](https://www.youtube.com/watch?v=pjxGPZQeeO4) by Google
* [Speaker Diarization: Optimal Clustering and Learning Speaker Embeddings](https://www.youtube.com/watch?v=vcyB8xb1-ys) by Microsoft Research
* [Robust Speaker Diarization for Meetings: the ICSI system](https://www.youtube.com/watch?v=kEcUcfLmIS0) by Microsoft Research

## Products

* [Google Cloud Speech-to-Text API](https://cloud.google.com/speech-to-text/docs/multiple-voices)
* [Amazon Transcribe](https://aws.amazon.com/transcribe/)
* [DeepAffects](https://www.deepaffects.com/diarization-api/)
