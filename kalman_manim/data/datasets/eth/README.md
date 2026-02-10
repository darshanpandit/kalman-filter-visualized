# ETH Pedestrian Dataset

Overhead-camera 2D pedestrian trajectories from ETH Zurich.

**Citation:**
Pellegrini, S., Ess, A., Schindler, K., & van Gool, L. (2009).
*You'll Never Walk Alone: Modeling Social Behavior for Multi-Target Tracking.*
IEEE International Conference on Computer Vision (ICCV).

**Source:** Preprocessed 4-column format from
[Social-STGCNN](https://github.com/abduallahmohamed/Social-STGCNN),
originally from the [ETH Walking Pedestrians dataset](https://icu.ee.ethz.ch/research/datsets.html).

**License:** BSD-style. Free for academic and non-commercial use.

## Format

Tab-separated, 4 columns per line:

```
frame_id    pedestrian_id    pos_x    pos_y
```

- Coordinates in meters (world frame, overhead camera homography)
- Frame IDs sampled every 10 original frames (~2.5 FPS at 25 FPS capture)

## Files

- `hotel.txt` — Hotel sequence (~389 pedestrians, 6543 observations)
- `eth.txt` — ETH/University sequence (~360 pedestrians, 5492 observations)
