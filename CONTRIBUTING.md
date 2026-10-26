# Contributing

Your contributions are always welcome!

Just send a pull request, and I will review and merge it.

## Guidance

### General

For new items, please follow the existing format.

### Publications

Only include complete papers that are directly related to speaker diarization.

For example, these will **NOT** be accepted:
* Course project reports.
* One-page descriptions of diarization systems submitted to the DIHARD challenge.
* Commercial system technical documents.
* Media posts.
* Low-quality papers without experiments and evaluations.
* Publications not directly related to speaker diarization:
  * Pure ML papers.
  * Speaker recognition papers.

### Software

* A **Framework** is a software toolkit that has all the necessary features to
  perform speaker diarization, including audio processing, feature extraction,
  speaker analysis and clustering, etc.
* An **Evaluation** package must be able to produce speaker-diarization-related
  metrics that are *permutation invariant*, such as Diarization Error Rate
  (DER).
* A **Clustering** library must correspond to a clustering algorithm that has
  been used by at least one diarization publication.
* Open-source end-user applications and auxiliary utilities belong in **Other
  software**, whereas commercial APIs and hosted SaaS offerings belong in
  **Products**.

### Datasets

A diarization dataset must contain utterances with multiple speakers
**speaking in turn**, and each utterance must have
**time-stamped speaker annotations**.
