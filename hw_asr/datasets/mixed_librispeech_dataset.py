import json
import logging
import os
import shutil
from pathlib import Path

import torchaudio
from speechbrain.utils.data_utils import download_file
from tqdm import tqdm

from hw_asr.base.base_dataset import BaseDataset
from hw_asr.utils import ROOT_PATH
from mixed_generator import MixtureGenerator, LibriSpeechSpeakerFiles

logger = logging.getLogger(__name__)

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechMixedDataset(BaseDataset):
    def __init__(
            self, part, data_dir=None, mixed_dir=None, data_write_dir=None, 
            snr_levels=[-5, 5], num_workers=5, mixer_audio_length=3, test=False, *args, **kwargs
            ):
        assert part in URL_LINKS or part == 'train_all'

        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "librispeech"
            data_dir.mkdir(exist_ok=True, parents=True)
        
        if mixed_dir is None:
            data_mixed_dir = ROOT_PATH / "data" / "datasets" / "librispeech_mixes"
            data_mixed_dir.mkdir(exist_ok=True, parents=True)

        self._data_dir = Path(data_dir)
        self._data_mixed_dir = Path(data_mixed_dir)
        self._data_write_dir = Path(data_write_dir) if data_write_dir else Path(data_dir)

        self.snr_levels = snr_levels
        self.num_workers = num_workers
        self.mixer_audio_length = mixer_audio_length
        self.test = test

        # {"ref": [ref_paths], "mix": [mix_paths], "target": [target_paths]}
        self.mixed_data_paths = None
        self.num_speakers = None

        if part == 'train_all':
            index = sum([self._get_or_load_index(part)
                         for part in URL_LINKS if 'train' in part], [])
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)


    def __getitem__(self, ind):
        data_dict = self._index[ind]

        item = {}
        for key in ["ref", "mix", "target"]:
            audio_path = data_dict[f"{key}_path"]
            audio_wave = self.load_audio(audio_path)
            audio_wave, audio_spec = self.process_wave(audio_wave)

            item[f"{key}_audio"] = audio_wave
            item[f"{key}_spectrogram"] = audio_spec
            item[f"{key}_path"] = audio_path
            item[f"{key}_duration"] = audio_wave.size(1) / self.config_parser["preprocessing"]["sr"]

        item["speaker_id"] = int(data_dict["ref_path"].split(' ')[0])

        return item


    def _load_part(self, part):
        arch_path = self._data_dir / f"{part}.tar.gz"
        print(f"Loading part {part}")
        download_file(URL_LINKS[part], arch_path)
        shutil.unpack_archive(arch_path, self._data_dir)
        for fpath in (self._data_dir / "LibriSpeech").iterdir():
            shutil.move(str(fpath), str(self._data_dir / fpath.name))
        os.remove(str(arch_path))
        shutil.rmtree(str(self._data_dir / "LibriSpeech"))
    

    def _load_mixed_part(self, part, nfiles=None, test=False):
        original_path = self._data_dir / part
        speaker_ids = [el.name for el in os.scandir(original_path)]

        mixed_path = self._data_mixed_dir / part
        speakers_files = [
            LibriSpeechSpeakerFiles(id, original_path, audioTemplate="*.flac")
            for id in speaker_ids
        ]

        if not nfiles:
            nfiles = len(speakers_files)

        generator = MixtureGenerator(speakers_files, mixed_path, nfiles, test)
        generator.generate_mixes(
            self.snr_levels, self.num_workers,
            update_steps=100, audioLen=self.mixer_audio_length
        )

    def _get_or_load_index(self, part):
        index_path = self._data_write_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_mixed_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
    
    def _create_mixed_index(self, part):
        index = []

        split_dir = self._data_dir / part
        if not split_dir.exists():
            self._load_part(part)
        
        split_mixed_dir = self._data_mixed_dir / part
        if not split_mixed_dir.exists():
            self._load_mixed_part(part, test=self.test)
            # {"ref": [ref_paths], "mix": [mix_paths], "target": [target_paths]}
            self.mixed_data_paths = self.generator.sort_mixes(split_mixed_dir)

            speaker_ids = set()
            for path in self.mixed_data_paths:
                if path[-7:] == "ref.wav":
                    speaker_ids.add(int(path.split(' ')[0]))
            self.num_speakers = len(speaker_ids)

        ref_paths = self.mixed_data_paths["ref"]
        mix_paths = self.mixed_data_paths["mix"]
        target_paths = self.mixed_data_paths["target"]

        for ref_path, mix_path, target_path in tqdm(
                zip(ref_paths, mix_paths, target_paths),
                desc=f"Preparing librispeech mixes: {part}"
        ):
            index.append({
                "ref_path": ref_path,
                "mix_path": mix_path,
                "target_path": target_path
            })

        return index
