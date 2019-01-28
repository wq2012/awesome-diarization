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
* One pager description of a diarization system submitted to DIHARD challenge.
* Commercial system technical document.
* Media post.
* Low quality paper without experiments and evaluations.
* Publications not directly related to spealer diarization:
  * Pure ML papers.
  * Speaker recognition papers.

### Software

* A **Framework** is a software that has all the necessary features to perform
  speaker diarization, including audio processing, feature extraction,
  speaker analysis and clustering, etc.
* A **Evaluation** software must be able to produce speaker diarization related
  metrics that are *permutation invariant*, such as Diarization Error Rate
  (DER).
* A **Clustering** software must correspond to a clustering algorithm that has
  been used by at least one diarization publication.

### Dataset

A dataset must contain utterances with multiple speakers, and each utterance
must have time-stamped speaker annotations.
