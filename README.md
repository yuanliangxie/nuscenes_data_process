# A data processing mechanism for radar and camera fusion
It's the official code for the paper `A Feature Pyramid Fusion Detection Algorithm Based on Radar and Camera Sensor`.
The code is used to processing nuScenes dataset and generating the radar projection map with line render shape or circle render shape for 2D fusion detection.The dataloader has been implement by Pytorch dataloader module.So you can easily to use the data processing module to your fusion-net.

If this open source code can help your research, it is my honor!

###How to start:
- first download the v1.0-mini dataset to have a try!
- edit `./train/get_convert_sample_token.py` and change the 'dataroot' to  your v1.0-mini's root
 ```
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/xyl/PycharmProjects/crfnet-comback/v1.0-mini', verbose=True)
```
- edit `./train/params_init_nuScenes.py` for your adapting set.
- edit `radar_fusion_generator/fusion_generator` in the last. it's also change
the `dataroot` for dataset.
- ```
  cd ./train
  python get_convert_sample_token.py
  cd ../radar_fusion_generator
  python fusion_generator.py
  ```
  Then you can see the radar projection map!



The code borrorws some codes from the [CRF-net](https://github.com/TUMFTM/CameraRadarFusionNet).
Thanks for their enlightening work.



