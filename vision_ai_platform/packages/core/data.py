from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class BaseTransform:
    """Base class for image transformations

    This class provides a unified interface for applying transformations to images, object instances, and semantic
    segmentation masks. Subclasses should override `apply_image`, `apply_instances`, and/or `apply_semantic` for simple
    transforms, or override `__call__` directly for complex transforms that need shared state between image and
    annotation modifications.

    Methods:
        get_params: Compute transformation parameters shared across image, instances, and semantic mask.
        apply_image: Apply transformation to the image in labels['img'].
        apply_instances: Apply transformation to object instances in labels['instances'].
        apply_semantic: Apply transformation to semantic mask in labels['semantic_mask'].
        __call__: Orchestrate the transformation pipeline.
    """

    def __call__(self, labels):
        """Apply transformation to labels dict.

        Args:
            labels (dict): Dictionary containing 'img', optionally 'instances' and 'semantic_mask'.

        Returns:
            (dict): Transformed labels dictionary.
        """
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        return labels

    def get_params(self, labels):
        """Compute and return transformation parameters.

        This method allows sharing random state or computed matrices (e.g. affine matrix, flip
        decision) between image, instances, and semantic mask transformations.

        Args:
            labels (dict): Input labels dictionary.

        Returns:
            (dict): Parameters to pass to apply_image, apply_instances, and apply_semantic.
        """
        return {}

    def apply_image(self, labels, params=None):
        """Apply transformation to image.

        Args:
            labels (dict): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels

    def apply_instances(self, labels, params=None):
        """Apply transformation to object instances.

        Args:
            labels (dict): Dictionary containing 'instances'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels

    def apply_semantic(self, labels, params=None):
        """Apply transformation to semantic segmentation mask.

        Args:
            labels (dict): Dictionary containing 'semantic_mask'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels

    def apply_depth(self, labels, params=None):
        """Apply transformation to depth map.

        Args:
            labels (dict): Dictionary containing 'depth'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels dictionary.
        """
        return labels


class Compose:
    """A class for composing multiple image transformations.

    Attributes:
        transforms (list[Callable]): A list of transformation functions to be applied sequentially.

    Methods:
        __call__: Apply a series of transformations to input data.
        append: Append a new transform to the existing list of transforms.
        insert: Insert a new transform at a specified index in the list of transforms.
        __getitem__: Retrieve a specific transform or a set of transforms using indexing.
        __setitem__: Set a specific transform or a set of transforms using indexing.
        tolist: Convert the list of transforms to a standard Python list.

    Examples:
        >>> transforms = [RandomFlip(), RandomPerspective(30)]
        >>> compose = Compose(transforms)
        >>> transformed_data = compose(data)
        >>> compose.append(CenterCrop((224, 224)))
        >>> compose.insert(0, RandomFlip())
    """

    def __init__(self, transforms):
        """Initialize the Compose object with a list of transforms.

        Args:
            transforms (list[Callable]): A list of callable transform objects to be applied sequentially.
        """
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Apply a series of transformations to input data.

        This method sequentially applies each transformation in the Compose object's transforms to the input data.

        Args:
            data (Any): The input data to be transformed. This can be of any type, depending on the transformations in
                the list.

        Returns:
            (Any): The transformed data after applying all transformations in sequence.

        Examples:
            >>> transforms = [Transform1(), Transform2(), Transform3()]
            >>> compose = Compose(transforms)
            >>> transformed_data = compose(input_data)
        """
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Append a new transform to the existing list of transforms.

        Args:
            transform (BaseTransform): The transformation to be added to the composition.

        Examples:
            >>> compose = Compose([RandomFlip(), RandomPerspective()])
            >>> compose.append(RandomHSV())
        """
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Insert a new transform at a specified index in the existing list of transforms.

        Args:
            index (int): The index at which to insert the new transform.
            transform (BaseTransform): The transform object to be inserted.

        Examples:
            >>> compose = Compose([Transform1(), Transform2()])
            >>> compose.insert(1, Transform3())
            >>> len(compose.transforms)
            3
        """
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> Compose:
        """Retrieve a specific transform or a set of transforms using indexing.

        Args:
            index (int | list[int]): Index or list of indices of the transforms to retrieve.

        Returns:
            (Compose | Any): A new Compose object if index is a list, or a single transform if index is an int.

        Raises:
            AssertionError: If the index is not of type int or list.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), RandomHSV(0.5, 0.5, 0.5)]
            >>> compose = Compose(transforms)
            >>> single_transform = compose[1]  # Returns the RandomPerspective transform directly
            >>> multiple_transforms = compose[[0, 1]]  # Returns a Compose object with RandomFlip and RandomPerspective
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]

    def __setitem__(self, index: list | int, value: list | int) -> None:
        """Set one or more transforms in the composition using indexing.

        Args:
            index (int | list[int]): Index or list of indices to set transforms at.
            value (Any | list[Any]): Transform or list of transforms to set at the specified index(es).

        Raises:
            AssertionError: If index type is invalid, value type doesn't match index type, or index is out of range.

        Examples:
            >>> compose = Compose([Transform1(), Transform2(), Transform3()])
            >>> compose[1] = NewTransform()  # Replace second transform
            >>> compose[[0, 1]] = [NewTransform1(), NewTransform2()]  # Replace first two transforms
        """
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
            )
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """Convert the list of transforms to a standard Python list.

        Returns:
            (list): A list containing all the transform objects in the Compose instance.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(10), CenterCrop()]
            >>> compose = Compose(transforms)
            >>> transform_list = compose.tolist()
            >>> print(len(transform_list))
            3
        """
        return self.transforms

    def __repr__(self):
        """Return a string representation of the Compose object.

        Returns:
            (str): A string representation of the Compose object, including the list of transforms.

        Examples:
            >>> transforms = [RandomFlip(), RandomPerspective(degrees=10, translate=0.1, scale=0.1)]
            >>> compose = Compose(transforms)
            >>> print(compose)
            Compose([
                RandomFlip(),
                RandomPerspective(degrees=10, translate=0.1, scale=0.1)
            ])
        """
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform(BaseTransform):
    """Base class for mix transformations like Cutmix, MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the probability-based
    application of transforms and manages the mixing of multiple images and labels.

    Attributes:
        dataset (Any): The dataset object containing images and labels.
        pre_transform (Callable | None): Optional transform to apply before mixing.
        p (float): Probability of applying the mix transformation.

    Methods:
        __call__: Apply the mix transformation to the input labels.
        get_params: Prepare mixed labels and update text labels.
        get_indexes: Abstract method to get indexes of images to be mixed.
        _update_label_text: Update label text for mixed images.

    Examples:
        >>> class CustomMixTransform(BaseMixTransform):
        ...     def apply_image(self, labels, params=None):
        ...         # Implement custom image mixing here
        ...         return labels
        ...
        ...     def get_indexes(self):
        ...         return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        >>> dataset = YourDataset()
        >>> transform = CustomMixTransform(dataset, p=0.5)
        >>> mixed_labels = transform(original_labels)
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initialize the BaseMixTransform object for mix transformations like CutMix, MixUp and Mosaic.

        This class serves as a base for implementing mix transformations in image processing pipelines.

        Args:
            dataset (Any): The dataset object containing images and labels for mixing.
            pre_transform (Callable | None): Optional transform to apply before mixing.
            p (float): Probability of applying the mix transformation. Should be in the range [0.0, 1.0].
        """
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Apply pre-processing transforms and cutmix/mixup/mosaic transforms to labels data.

        This method determines whether to apply the mix transform based on a probability factor. If applied, it selects
        additional images, applies pre-transforms if specified, and then performs the mix transform.

        Args:
            labels (dict[str, Any]): A dictionary containing label data for an image.

        Returns:
            (dict[str, Any]): The transformed labels dictionary, which may include mixed data from other images.

        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels

        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        labels.pop("mix_labels", None)
        return labels

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Prepare mixed labels and update text labels.

        Args:
            labels (dict[str, Any]): A dictionary containing label data for an image.

        Returns:
            (dict[str, Any]): Parameters for apply_image, apply_instances, and apply_semantic.
        """
        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic, CutMix or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        self._update_label_text(labels)
        return {"mix_labels": mix_labels}

    def get_indexes(self):
        """Get a random index for mosaic augmentation.

        Returns:
            (int): A random index from the dataset.

        Examples:
            >>> transform = BaseMixTransform(dataset)
            >>> index = transform.get_indexes()
            >>> print(index)  # 7
        """
        return random.randint(0, len(self.dataset) - 1)

    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        """Update label text and class IDs for mixed labels in image augmentation.

        This method processes the 'texts' and 'cls' fields of the input labels dictionary and any mixed labels, creating
        a unified set of text labels and updating class IDs accordingly.

        Args:
            labels (dict[str, Any]): A dictionary containing label information, including 'texts' and 'cls' fields, and
                optionally a 'mix_labels' field with additional label dictionaries.

        Returns:
            (dict[str, Any]): The updated labels dictionary with unified text labels and updated class IDs.

        Examples:
            >>> labels = {
            ...     "texts": [["cat"], ["dog"]],
            ...     "cls": torch.tensor([[0], [1]]),
            ...     "mix_labels": [{"texts": [["bird"], ["fish"]], "cls": torch.tensor([[0], [1]])}],
            ... }
            >>> updated_labels = BaseMixTransform._update_label_text(labels)
            >>> print(updated_labels["texts"])
            [['cat'], ['dog'], ['bird'], ['fish']]
            >>> print(updated_labels["cls"])
            tensor([[0],
                    [1]])
            >>> print(updated_labels["mix_labels"][0]["cls"])
            tensor([[2],
                    [3]])
        """
        if "texts" not in labels:
            return labels

        mix_texts = [*labels["texts"], *(item for x in labels["mix_labels"] for item in x["texts"])]
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
    augmentation is applied to a dataset with a given probability.

    Attributes:
        dataset: The dataset on which the mosaic augmentation is applied.
        imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
        p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
        n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        border (tuple[int, int]): Border size for height and width.

    Methods:
        get_indexes: Return a list of random indexes from the dataset.
        get_params: Compute mosaic layout parameters.
        apply_image: Allocate canvas and paste images into mosaic.
        apply_instances: Concatenate and clip instances for mosaic.
        _update_labels: Update labels with padding.
        _cat_labels: Concatenate labels and clips mosaic border instances.

    Examples:
        >>> from ultralytics.data.augment import Mosaic
        >>> dataset = YourDataset(...)  # Your image dataset
        >>> mosaic_aug = Mosaic(dataset, imgsz=640, p=0.5, n=4)
        >>> augmented_labels = mosaic_aug(original_labels)
    """

    def __init__(self, dataset, imgsz: int = 640, p: float = 1.0, n: int = 4):
        """Initialize the Mosaic augmentation object.

        This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image. The
        augmentation is applied to a dataset with a given probability.

        Args:
            dataset (Any): The dataset on which the mosaic augmentation is applied.
            imgsz (int): Image size (height and width) after mosaic pipeline of a single image.
            p (float): Probability of applying the mosaic augmentation. Must be in the range 0-1.
            n (int): The grid size, either 4 (for 2x2) or 9 (for 3x3).
        """
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n
        self.buffer_enabled = self.dataset.cache != "ram"

    def get_indexes(self):
        """Return a list of random indexes from the dataset for mosaic augmentation.

        This method selects random image indexes either from a buffer or from the entire dataset, depending on the
        'buffer_enabled' attribute. It is used to choose images for creating mosaic augmentations.

        Returns:
            (list[int]): A list of random image indexes. The length of the list is n-1, where n is the number of images
                used in the mosaic (either 3 or 8, depending on whether n is 4 or 9).

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640, p=1.0, n=4)
            >>> indexes = mosaic.get_indexes()
            >>> print(len(indexes))  # Output: 3
        """
        if self.buffer_enabled:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
        """Compute mosaic layout parameters.

        Args:
            labels (dict[str, Any]): Input labels dictionary.

        Returns:
            (dict[str, Any]): Parameters including 'layout' with per-patch geometry.
        """
        params = super().get_params(labels)
        assert labels.get("rect_shape") is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."

        s = self.imgsz
        layout = []
        if self.n == 4:
            yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
            for i in range(4):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # top left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                elif i == 1:  # top right
                    x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                    x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
                elif i == 2:  # bottom left
                    x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
                elif i == 3:  # bottom right
                    x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                    x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
                padw = x1a - x1b
                padh = y1a - y1b
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1a": x1a,
                        "y1a": y1a,
                        "x2a": x2a,
                        "y2a": y2a,
                        "x1b": x1b,
                        "y1b": y1b,
                        "x2b": x2b,
                        "y2b": y2b,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
        elif self.n == 9:
            hp, wp = -1, -1
            h0, w0 = None, None
            for i in range(9):
                labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
                img = labels_patch["img"]
                h, w = labels_patch.get("resized_shape", img.shape[:2])
                if i == 0:  # center
                    c = s, s, s + w, s + h
                    h0, w0 = h, w
                elif i == 1:  # top
                    c = s, s - h, s + w, s
                elif i == 2:  # top right
                    c = s + wp, s - h, s + wp + w, s
                elif i == 3:  # right
                    c = s + w0, s, s + w0 + w, s + h
                elif i == 4:  # bottom right
                    c = s + w0, s + hp, s + w0 + w, s + hp + h
                elif i == 5:  # bottom
                    c = s + w0 - w, s + h0, s + w0, s + h0 + h
                elif i == 6:  # bottom left
                    c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
                elif i == 7:  # left
                    c = s - w, s + h0 - h, s, s + h0
                elif i == 8:  # top left
                    c = s - w, s + h0 - hp - h, s, s + h0 - hp
                padw, padh = c[:2]
                x1, y1, x2, y2 = (max(x, 0) for x in c)
                layout.append(
                    {
                        "labels_patch": labels_patch,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "padw": padw,
                        "padh": padh,
                        "img_shape": (h, w),
                    }
                )
                hp, wp = h, w
        params["layout"] = layout
        return params

    def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply mosaic augmentation to the image.

        Args:
            labels (dict[str, Any]): Dictionary containing 'img'.
            params (dict | None): Parameters from get_params, including 'layout'.

        Returns:
            (dict): Updated labels with mosaic image.
        """
        layout = params["layout"]
        if self.n == 4:
            img4 = np.full((self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img4
        elif self.n == 9:
            img9 = np.full((self.imgsz * 3, self.imgsz * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                img = labels_patch["img"]
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                img9[y1:y2, x1:x2] = img[y1b:y2b, x1b:x2b]
            labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    def apply_instances(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply mosaic augmentation to instances.

        Args:
            labels (dict[str, Any]): Dictionary containing 'instances' and 'cls'.
            params (dict | None): Parameters from get_params, including 'layout'.

        Returns:
            (dict): Updated labels with concatenated instances.
        """
        layout = params["layout"]
        mosaic_labels = []
        for item in layout:
            if self.n == 4:
                padw = item["padw"]
                padh = item["padh"]
            else:  # n == 9
                padw = item["padw"] + self.border[0]
                padh = item["padh"] + self.border[1]
            labels_patch = self._update_labels(item["labels_patch"], padw, padh, item.get("img_shape"))
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        labels.update(final_labels)
        return labels

    def apply_semantic(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply mosaic augmentation to semantic mask.

        Args:
            labels (dict[str, Any]): Dictionary containing 'semantic_mask'.
            params (dict | None): Parameters from get_params.

        Returns:
            (dict): Updated labels with concatenated semantic mask.
        """
        if labels.get("semantic_mask") is None and all(
            m.get("semantic_mask") is None for m in labels.get("mix_labels", [])
        ):
            return labels

        layout = params["layout"]
        if self.n == 4:
            mask4 = np.full((self.imgsz * 2, self.imgsz * 2), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
                x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
                mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask4
        elif self.n == 9:
            mask9 = np.full((self.imgsz * 3, self.imgsz * 3), 255, dtype=np.uint8)
            for item in layout:
                labels_patch = item["labels_patch"]
                mask = labels_patch.get("semantic_mask")
                if mask is None:
                    continue
                x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
                padw, padh = item["padw"], item["padh"]
                x1b, y1b = x1 - padw, y1 - padh
                x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
                mask9[y1:y2, x1:x2] = mask[y1b:y2b, x1b:x2b]
            labels["semantic_mask"] = mask9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
        return labels

    @staticmethod
    def _update_labels(labels, padw: int, padh: int, img_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Update label coordinates with padding values.

        This method adjusts the bounding box coordinates of object instances in the labels by adding padding
        values. It also denormalizes the coordinates if they were previously normalized.

        Args:
            labels (dict[str, Any]): A dictionary containing image and instance information.
            padw (int): Padding width to be added to the x-coordinates.
            padh (int): Padding height to be added to the y-coordinates.
            img_shape (tuple[int, int] | None): Optional (h, w) of the original patch image. Needed because apply_image
                may overwrite labels["img"] with the mosaic canvas before apply_instances runs.

        Returns:
            (dict): Updated labels dictionary with adjusted instance coordinates.

        Examples:
            >>> labels = {"img": np.zeros((100, 100, 3)), "instances": Instances(...)}
            >>> padw, padh = 50, 50
            >>> updated_labels = Mosaic._update_labels(labels, padw, padh)
        """
        nh, nw = img_shape if img_shape is not None else labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels: list[dict[str, Any]]) -> dict[str, Any]:
        """Concatenate and process labels for mosaic augmentation.

        This method combines labels from multiple images used in mosaic augmentation, clips instances to the mosaic
        border, and removes zero-area boxes.

        Args:
            mosaic_labels (list[dict[str, Any]]): A list of label dictionaries for each image in the mosaic.

        Returns:
            (dict[str, Any]): A dictionary containing concatenated and processed labels for the mosaic image, including:
                - im_file (str): File path of the first image in the mosaic.
                - ori_shape (tuple[int, int]): Original shape of the first image.
                - resized_shape (tuple[int, int]): Shape of the mosaic image (imgsz * 2, imgsz * 2).
                - cls (np.ndarray): Concatenated class labels.
                - instances (Instances): Concatenated instance annotations.
                - texts (list[str], optional): Text labels if present in the original labels.

        Examples:
            >>> mosaic = Mosaic(dataset, imgsz=640)
            >>> mosaic_labels = [{"cls": np.array([0, 1]), "instances": Instances(...)} for _ in range(4)]
            >>> result = mosaic._cat_labels(mosaic_labels)
            >>> print(result.keys())
            dict_keys(['im_file', 'ori_shape', 'resized_shape', 'cls', 'instances'])
        """
        # if not mosaic_labels:
        #     return {}
        # cls = []
        # instances = []
        # imgsz = self.imgsz * 2  # mosaic imgsz
        # for labels in mosaic_labels:
        #     cls.append(labels["cls"])
        #     instances.append(labels["instances"])
        # # Final labels
        # final_labels = {
        #     "im_file": mosaic_labels[0]["im_file"],
        #     "ori_shape": mosaic_labels[0]["ori_shape"],
        #     "resized_shape": (imgsz, imgsz),
        #     "cls": np.concatenate(cls, 0),
        #     "instances": Instances.concatenate(instances, axis=0),
        # }
        # final_labels["instances"].clip(imgsz, imgsz)
        # good = final_labels["instances"].remove_zero_area_boxes()
        # final_labels["cls"] = final_labels["cls"][good]
        # if "texts" in mosaic_labels[0]:
        #     final_labels["texts"] = mosaic_labels[0]["texts"]
        # return final_labels