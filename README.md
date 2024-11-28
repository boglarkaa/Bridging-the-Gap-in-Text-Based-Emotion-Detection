# Bridging-the-Gap-in-Text-Based-Emotion-Detection
SemEval Task 11 - Bridging the Gap in Text-Based Emotion Detection

### Track A: Multi-label Emotion Detection
Given a target text snippet, predict the perceived emotion(s) of the speaker. Specifically, select whether each of the following emotions apply: joy, sadness, fear, anger, surprise, or disgust. In other words, label the text snippet with: joy (1) or no joy (0), sadness (1) or no sadness (0), anger (1) or no anger (0), surprise (1) or no surprise (0), and disgust (1) or no disgust (0).

Note that for some languages such as English, the set perceived emotions includes 5 emotions: joy, sadness, fear, anger, or surprise and does not include disgust.

A training dataset with gold emotion labels will be provided for this track.

Below is a sample of the English training data (Track A). A text snippet can have multiple emotions (e.g., the sentence with the ID sample_05 expresses both joy and surprise), or none (e.g., sample_04 with all the emotion values equal to 0 is considered neutral).
