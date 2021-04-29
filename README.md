# Discover computer vision


## Luxonis

```sh 
python3 -m venv .env.private
source .env.private/bin/activate

python3 -m pip install -r requirements.txt


ffmpeg -i Experiment\ 1\ v3\ -\ HD\ 720p.mov -vcodec libx265 -crf 18 experiment_1_v3.1.mp4
```

- https://docs.luxonis.com/projects/api/en/latest
- https://github.com/openvinotoolkit/open_model_zoo
- https://towardsdatascience.com/robot-tank-with-raspberry-pi-and-intel-neural-computer-stick-2-77263ca7a1c7


- [List of intel models](https://github.com/openvinotoolkit/open_model_zoo/tree/2019_R3/models/intel)
- https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public
- [Examples with luxonis](https://github.com/luxonis/depthai-experiments)
- https://docs.luxonis.com/en/latest/pages/training/

## Build models

```sh
$MYRIAD_COMPILE -m ~/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml -ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4


```
- https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/myriad-compile-dyld-Library-not-loaded-rpath-libtbb-dylib-on-Mac/td-p/1249039

