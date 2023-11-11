import torch
from torch import nn

from hw_asr.base import BaseModel

class SpExPlusModel(BaseModel):
    def __init__(
            self,
            L1, L2, L3,
            N_filters,

            **batch
    ):
        """
        Implementation of SpEx Plus Model for Speech Separation Task
        from the "SpEx+: A Complete Time Domain Speaker Extraction Network" paper.
        https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf

        The model itself encodes mixed and target-speaker waveforms into 
        the shared latent space and then uses target-speaker embedding 
        to predict the mask for the mixed audio. 

        Overall, architecture consists of:
        1) Twin Speech Encoders, which encode initial audios with 
            1d-Convolutions with 3 different filter lengths. 
            These layers provide model backbone with multiple scales of temporal resolutions.
        2) Speaker Encoder - this part encodes the speaker embedding 
            from the clean target audio, in order to help the main part of the model.
            Learning is performed both on the Cross-Entropy Loss for speaker classification
            and SI-SDR Loss backpropagated from the main backbone.
        3) Speaker Extractor - this part encodes the mixed audio, 
            adding speaker-embedding from Speaker Encoder part to its Stacked TCN Blocks.
            This part predicts the mask on the mixed encoded audio
        4) Decoders on 3 different scales - decode the masked mixed audio back,
            predicting the denoised audio of the target speaker.
            Its outputs are fed to the SI-SDR Loss, that trains the main model.

        To get the further understanding of model architecture, refer to the paper itself.

        params:
            L1, L2, L3: filter lengths of the Encoder Conv1D's
            N_filters: number of filters for each of L1, L2, L3
        """

        self.L1, self.L2, self.L3 = L1, L2, L3

        self.encoders = {
            filter: nn.Conv1D(
                in_channels=1,
                out_channels=N_filters,
                kernel_size=getattr(self, filter),
                stride=getattr(self, filter) // 2,
                padding=0
            )
            for filter in ["L1", "L2", "L3"]
        }
        self.encoder_layer_norm = nn.LayerNorm(3 * N_filters)



