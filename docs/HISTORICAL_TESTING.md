# Physical testing and labeled calibration

I developed the coaching rules by performing exercises, recording good and bad form, and replaying the saved body points through the Swift coaches. The recorded examples are all of me. Friends also tried the app informally to explore how it worked with different body proportions.

## Recorded-video tests

The original project contains 51 video files and 34 JSON replay fixtures, including one synthetic fixture. The Swift replay harness passes timestamped body points to production coaches and checks repetition counts, form violations, and coaching cues.

Examples include planks with raised or sagging hips, dead-bug exercises with legs resting or knees too bent, and squats with a rounded back or excessive forward lean. The test source contains 68 video-harness methods across ten files, covering checks and diagnostic output. These counts describe the recovered test material.

The standalone repository supplies the Python export tools. The recordings, personal motion records, and app's Swift test target remain in the private app project.

## Labeled pike repetitions

I compared 37 pike and knee-pike repetitions, with 22 labeled as good form and 15 as bad form. The historical analyzer summarizes measurement distributions from these labels.

| Exercise | Good form | Bad form | Median hip angle, good / bad |
| --- | --- | --- | --- |
| Knee pike | 10 | 8 | 137.7° / 97.7° |
| Full pike | 12 | 7 | 143.0° / 124.9° |

The stored trajectory measurements also showed that a rule developed from a controlled recording did not behave consistently in device testing. I revised the coaching logic to use head movement and hip angle, with separate settings for the two exercise variations. These records supported calibration of the rules.

For body-size normalization, the app expresses head movement relative to torso length. The Python metric calculator expresses hand spacing relative to shoulder width. These ratios help compare movement at different body sizes and camera distances.

## Running the analyzer

```sh
python analyze_rep_calibration.py /path/to/rep_calibration.jsonl
```

The historical schema uses the raw labels `good` and `bob`; in this document these correspond to good form and bad form. The command accepts the earlier scalar-label JSONL format. The later app schema uses different fields and needs an adapter.

The analyzer successfully processed the 37 saved records. The separate seven synthetic Python checks cover conversion, missing poses, timing, segmentation, and export; their execution is recorded in [VALIDATION.md](VALIDATION.md).
