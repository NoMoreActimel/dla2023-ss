# SS project

This is a repository with the SS homework of the HSE DLA Course. It includes the implementation of SpEx+ model architecture and all training utilities. The training was performed on the [LibriSpeech](https://www.openslr.org/12) train-clean-100 dataset, by artificially mixing pairs of audios.

## Installation guide

Clone the repository:
```shell
%cd local/cloned/project/path
git clone https://github.com/NoMoreActimel/dla2023-ss.git
```

Install and create pyenv:
```shell
pyenv install 3.9.7
cd path/to/local/cloned/project
~/.pyenv/versions/3.9.7/bin/python -m venv asr_venv
```

Install required packages:

```shell
pip install -r ./requirements.txt
```

You may now launch training / testing of the model, specifying the config file. The default model config is given as default_test_config.json. However, you may check for other examples in hw_asr/configs directory.

There is no good-enough pretrained model, as I was mainly debugging model architecture on the short-term launches. It appeared to be, that basically to achieve comparable perfomance you need to train 10s or 100s of times more, so I didn't made it till the deadline.


Overall, to launch pretrained model you need to download the [model-checkpoint](https://drive.google.com/drive/folders/1uE4WQs2Rjczn2t49ELcBMxjFijinG-Er?usp=sharing) and launch the test.py:
```shell
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1uE4WQs2Rjczn2t49ELcBMxjFijinG-Er
unzip checkpoint-best/checkpoint-best.zip checkpoint.pth
mv checkpoint.pth default_test_model/checkpoint.pth
mv checkpoint-best/config.json default_test_model/config.json
```
```shell
python test.py \
   -c default_test_config.json \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
``` 


## Structure

All written code is located in the hw_asr (heh) repository. Scripts launching training and testing are given as train.py and test.py in the root project directory. First one will call the trainer instance, which is the class used for training the model. Further on, trainer and base_trainer iterate over given datasets and log all provided metrics - main ones are SiSDR and PESQ from torchmetrics.audio. SiSDR has been also implemented manually and incorporated into the overall loss of the model. For the convenience everything is getting logged using the wandb logger, you may also look audios and many interesting model-weights graphs out there.

## Training

To train the model you need to specify the config path:
```shell
python3 train.py -c hw_asr/configs/config_name.json
```
If you want to proceed training process from the saved checkpoint, then:
```shell
python3 train.py -c hw_asr/configs/config_name.json -r saved/checkpoint/path.pth
```

## Testing

Some basic tests are located in hw_asr/tests directory. Script to run them:

```shell
python3 -m unittest discover hw_asr/tests
```

LibriSpeech test-clean results:

WER argmax: 0.24

CER argmax: 0.07

\
WER lm-beamsearch with beamwidth = 5: 0.18

CER lm-beamsearch with beamwidth = 5: 0.07
