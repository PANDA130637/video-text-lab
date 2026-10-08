# Video quality diagnostics

This report uses frozen CLIP frame/text features. The combined score is a transparent heuristic, not a human-calibrated perceptual metric.

- Videos: 27
- Mean quality score: 74.23 ± 0.51
- 95% bootstrap interval: 74.03–74.43
- Mean frame-text alignment: 0.2396
- Mean adjacent-frame cosine: 0.9397
- Mean flicker rate (threshold 0.85): 0.021

## Perturbation diagnostics

- clean: score 74.23, alignment 0.2396, adjacent cosine 0.9397, flicker rate 0.021
- shuffled_frames: score 74.19, alignment 0.2396, adjacent cosine 0.9376, flicker rate 0.011
- cross_video_flicker: score 73.06, alignment 0.2388, adjacent cosine 0.8743, flicker rate 0.339

## By semantic label

- boxing: 73.77 (95% CI 73.53–73.95), alignment 0.2244, flicker 0.016
- handclapping: 74.43 (95% CI 74.20–74.64), alignment 0.2400, flicker 0.000
- handwaving: 74.50 (95% CI 74.19–74.83), alignment 0.2544, flicker 0.048

## Limitations

The score can reward visually smooth but semantically wrong videos and can penalize intentional fast motion. Generated-video evaluation should be calibrated against human pairwise judgments before being used as a model-selection target.
