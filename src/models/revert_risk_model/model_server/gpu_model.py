import logging

import torch
from base_model import RevisionRevertRiskModel


class RevertRiskMultilingualGPU(RevisionRevertRiskModel):
    def __init__(
        self,
        name: str,
        module_name: str,
        model_path: str,
        wiki_url: str,
        aiohttp_client_timeout: int,
        force_http: bool,
    ) -> None:
        super().__init__(
            name,
            module_name,
            model_path,
            wiki_url,
            aiohttp_client_timeout,
            force_http,
        )
        self.use_gpu()

    def use_gpu(self):
        """
        Loads the model in the GPU's memory and updates its reference.
        This function needs to run after the webserver's initialization
        (that forks and creates new processes, see https://github.com/pytorch/pytorch/issues/83973).
        """
        if not self.device:
            self.device = torch.device("cuda")
            logging.info(f"Using device: {self.device}")
            # loading model to GPU
            self.model.title_model.model.to(self.device)
            self.model.insert_model.model.to(self.device)
            self.model.remove_model.model.to(self.device)
            self.model.change_model.model.to(self.device)
            # changing the device of the pipeline
            self.model.title_model.device = self.device
            self.model.insert_model.device = self.device
            self.model.remove_model.device = self.device
            self.model.change_model.device = self.device
