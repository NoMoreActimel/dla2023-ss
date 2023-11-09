# ASR project

This is a repository with the ASR homework of the HSE DLA Course. It includes the implementation of DeepSpeechV2 model architecture and all training utilities. The training was performed on the [LibriSpeech](https://www.openslr.org/12) dataset, train-clean-100/360 and train-other-500 in particular.

## Installation guide

Clone the repository:
```shell
%cd local/cloned/project/path
git clone https://github.com/NoMoreActimel/dla2023-asr.git
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

Run following commands to install beamseach language model:

```shell
wget https://openslr.elda.org/resources/11/4-gram.arpa.gz
gzip -d 4-gram.arpa.gz
mkdir -p data/librispeech-lm/
mv 4-gram.arpa data/librispeech-lm/
```

If you are specifiying external datasets in your config file and they are available in read-only mode, then you need to create index-directory for the internal dataset processing:
```shell
mkdir -p data/librispeech-index
```

You may now launch training / testing of the model, specifying the config file. The default model config is given as default_test_config.json. However, you may check for other examples in hw_asr/configs directory.

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

All written code is located in the hw_asr repository. Scripts launching training and testing are given as train.py and test.py in the root project directory. First one will call the trainer instance, which is the class used for training the model. Further on, trainer and base_trainer iterate over given datasets and log all provided metrics - main ones are Word Error Rate and Character Error Rate. For the convenience everything is getting logged using the wandb logger, you may also look for spectrograms, audios and many interesting model-weights graphs out there.

In order to improve the WER/CER metrics further, basical beamsearch algorithm has been implemented. However, the supported version of beam-search with language model guidance from pyctcdecode library works faster and generally significantly better. You can find both beamsearches in hw_asr/text_encoder/ctc_char_text_encoder.py

Due to the computational cost and time limitations, DeepSpeechV2 architecture was used with 5 Bidirectional-LSTM layers and hidden_size=512 (which is doubled to 1024 in Bidirectional-LSTM).


## Training

To train the model you need to specify the config path:
```shell
python3 train.py -c hw_asr/configs/config_name.json
```
If you want to proceed training process from the saved checkpoint, then:
```shell
python3 train.py -c hw_asr/configs/config_name.json -p saved/checkpoint/path.pth
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
