#from __future__ import annotations
from torch.utils.data import Dataset
from os import path, listdir
from typing import Optional, Tuple
from functools import reduce
from PIL import Image
import numpy as np


class ImageNetDataset(Dataset):
    """ImageNet 2012 Dataset
    
    Structure
    ---------
    train
        n01440764
        n01443537
        n01484850
        ...
    val
        n01440764
        n01443537
        n01484850
        ...
    labels.txt: txt file with 1000 mapped folder and class names.
    """
    
    def __init__(self, root_dir: str, phase: str, transforms: Optional[str]=None) -> None:
        """
        Set instance attributes.

        Parameters
        ----------
        root_dir:
            Path to the the dataset folder.
        phase:
            Dataset phase. Value must be train or val.
        transforms:
            Transforms for a input sample.

        Returns
        -------
        None
        """

        assert phase in ("train", "val")

        self.root_dir = root_dir
        self.phase = phase
        self.transforms = transforms

        self.classid_to_label, self._folder_stats, self._folder_names = self._load_annotations()
        self._dataset_len = self._get_len()

    def _load_annotations(self) -> Tuple[list, dict, dict]:
        """
        Returns indoramtion about data in dataset.

        Parameters
        ----------

        Returns
        -------
        class_to_label:
            Labels dictionary. Each class id mapped to class name.
        folder_stats:
            Folders dictionary. Each folder by name is mapped to class_id, folder size and images names.
        folder_names:
            List with folders names.
        """

        folder_names = sorted(listdir(path.join(self.root_dir, self.phase)),
                              key= lambda folder_name: int(folder_name[1:]))
        classid_to_label = {}
        folder_stats = {}

        with open(path.join(self.root_dir, 'labels.txt')) as f:
            for line in f:
                folder_name, id_, label_name = line.strip().split()
                classid_to_label[id_] = label_name
                list_dir = listdir(path.join(self.root_dir, self.phase, folder_name))
                folder_stats[folder_name] = {"id": int(id_),
                                             "len": len(list_dir),
                                             "images": sorted(list_dir, 
                                                       key=lambda image_name: int(image_name.replace('.', '_').split('_')[1]))}
        
        return classid_to_label, folder_stats, folder_names

    def _get_len(self) -> int:
        """
        Returns dataset len according to phase.

        Parameters
        ----------

        Returns
        -------
        len:
            Dataset len.
        """
        
        if self.phase == "val":
            return 50 * len(self._folder_stats) #each val folder consists of 50 images
        elif self.phase == "train":
            return reduce(lambda x, y: x + y, [self._folder_stats[folder]["len"] for folder in self._folder_names])

    def __len__(self) -> int:
        """
        Len magic method.

        Parameters
        ----------

        Returns
        -------
        len:
            Dataset len.
        """

        return self._dataset_len

    def __getitem__(self, global_idx: int) -> dict:
        """
        Returns item from dataset according to the phase.

        Parameters
        ----------
        global_idx:
            Item index in whole dataset.
        
        Returns
        -------
        item:
            Dataset item. Each item is a {"image": image, "class_id": class_id} dictionary.
        """

        folder_name, image_name = self._calculate_image_folder_and_name(global_idx)

        pillow_image = Image.open(path.join(self.root_dir, self.phase, folder_name, image_name))
        image = np.array(pillow_image, dtype = np.uint8)
        if len(image.shape) == 2:
            image = np.dstack((image, image, image))
        
        if self.transforms:
            image = self.transforms(image=image)["image"]

        return {"image": image,
                "class_id": self._folder_stats[folder_name]["id"]}

    def _calculate_image_folder_and_name(self, global_idx: int) -> Tuple[str, int]:
        """ Get folder by global index and local index in this folder.

        Parameters
        ----------
        global_idx: int
            Item index.

        Returns
        -------
        folder: str
            Image folder in dataset structure according to global index.
        local_idx: int
            Image index in folder. 
        """

        #self.folder_name is a sorted list
        if self.phase == "val":
            # 50 is an images number inside each val folder
            folder_name = self._folder_names[global_idx // 50]
            local_idx = global_idx % 50
        elif self.phase == "train":
            folder_i = global_idx // 1300 # first_approximation_folder_idx. 1300 is a maximum train folder size
            sum_ = 1300*folder_i

            while sum_ + self._folder_stats[self._folder_names[folder_i]]["len"] <= global_idx:
                sum_ += self._folder_stats[self._folder_names[folder_i]]["len"]
                folder_i += 1
            folder_name = self._folder_names[folder_i]
            local_idx = global_idx - sum_
        
        return folder_name, self._folder_stats[folder_name]["images"][local_idx]
