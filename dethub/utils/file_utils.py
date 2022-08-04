import urllib.request
from os import path
from pathlib import Path
from typing import Optional
import yaml

class FileUtils:
    def __init__(self, yaml_path: str, model_name: str, destination_path: str):
        self.yaml_path = yaml_path
        self.model_name = model_name
        self.destination_path = destination_path
        

    @staticmethod
    def download_from_url(from_url: str, to_path: str):

        Path(to_path).parent.mkdir(parents=True, exist_ok=True)

        if not path.exists(to_path):
            urllib.request.urlretrieve(
                from_url,
                to_path,
            )
    
    def model_download(self):
        import yaml
        with open(self.yaml_path, "r") as stream:
            self.model_name = yaml.safe_load(stream)["model_name"]
        
        if self.destination_path is None:
            self.destination_path = self.model_name[1]
        
        Path(self.destination_path).parent.mkdir(parents=True, exist_ok=True)
        if not path.exists(self.destination_path):
            urllib.request.urlretrieve(
                self.model_name[0],
                self.destination_path,
            )       
            