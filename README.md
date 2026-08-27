# Real-Time Facial Recognition System
> OpenCV pipeline from dataset capture to live webcam recognition, with a documented ROC evaluation.
`2025` · `Python` · `OpenCV` · `LBPH` · `Raspberry Pi 4` · `Computer vision`

![Real-Time Facial Recognition System](docs/img/face-live.jpg)

## About

A three-stage recognition pipeline built for a Raspberry Pi 4 target. Stage one captures and labels a face dataset from the camera; stage two trains a local recogniser and serialises the model and label map; stage three runs live inference on the webcam feed, drawing each detected face with its predicted identity and confidence score.

Recognition uses a confidence threshold — below it the face is accepted as a known identity, above it it is reported as unknown — and each label is drawn in its own colour. The application supports enrolling a new person without restarting and capturing annotated screenshots for evidence.

The classifier was evaluated with a documented ROC analysis rather than a single accuracy figure, which is what makes the confidence threshold a defensible choice instead of a guess.

## Figures

![face-pipeline.svg](docs/img/face-pipeline.svg)

## Contents

```
README.md
RECO.pdf
Untitled.ipynb
docs/
face__roc.docx
face__roc.pdf
face_dataset.py
face_recognition_app.py
face_train.py
raspberry-pi-4.png
```

## Notes

The training DATASET is deliberately excluded: it is photographs of real people and does not belong in a public repository. `face_dataset.py` rebuilds it from a webcam.

Not included in this repository: 2 belongs to another project file(s), 2 duplicate copy file(s), 3 excluded by name file(s) - build caches, generated toolpaths and oversized binaries are kept out on purpose. The source they are generated from is here.

## Third-party work used here

Everything in this repository is my own work. It builds on the following, which are **not** mine and are used under their own licences:

- **OpenCV (LBPH face recogniser, Haar cascades)** by OpenCV team — <https://opencv.org>

## Author

Lassaad Mahmoudi — <assaadmahmoudi0@gmail.com>  
https://linkedin.com/in/mahmoudiassaad
