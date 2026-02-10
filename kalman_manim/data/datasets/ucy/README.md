# UCY Pedestrian Dataset

Overhead-camera 2D pedestrian trajectories from the University of Cyprus.

**Citation:**
Lerner, A., Chrysanthou, Y., & Lischinski, D. (2007).
*Crowds by Example.*
Computer Graphics Forum, 26(3), 655-664.

**Source:** Preprocessed 4-column format from
[Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN),
originally from the [UCY Crowd Data](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data).

**License:** Free for academic and non-commercial use.

## Format

Tab-separated, 4 columns per line:

```
frame_id    pedestrian_id    pos_x    pos_y
```

- Coordinates in meters (world frame, overhead camera homography)
- Frame IDs sampled every 10 original frames (~2.5 FPS at 25 FPS capture)

## Files

- `univ.txt` — University/Students001 sequence (~390 pedestrians, 21813 observations)
- `zara1.txt` — Zara store sequence 01 (~148 pedestrians, 5153 observations)
- `zara2.txt` — Zara store sequence 02 (~204 pedestrians, 9722 observations)
