# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
import logging
import os
import collections
import copy
import torch
import torch.nn as nn
from termcolor import colored
from torch.nn.parallel import DataParallel, DistributedDataParallel
from typing import Any
from collections import defaultdict, OrderedDict


class Checkpointer(object):
    """
    A checkpointer that can save/load model as well as extra checkpointable
    objects.
    """

    def __init__(
        self,
        model: nn.Module,
        save_dir: str = "",
        *,
        save_to_disk: bool = True,
        **checkpointables: object,
    ):
        """
        Args:
            model (nn.Module): model.
            save_dir (str): a directory to save and find checkpoints.
            save_to_disk (bool): if True, save checkpoint to disk, otherwise
                disable saving for this checkpointer.
            checkpointables (object): any checkpointable objects, i.e., objects
                that have the `state_dict()` and `load_state_dict()` method. For
                example, it can be used like
                `Checkpointer(model, "dir", optimizer=optimizer)`.
        """
        if isinstance(model, (DistributedDataParallel, DataParallel)):
            model = model.module
        self.model = model
        self.checkpointables = copy.copy(checkpointables)
        self.logger = logging.getLogger('Context-Transformer.' + __name__)
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk

    def save(self, name: str, **kwargs: dict):
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = dict()
        data["model"] = self.model.state_dict()
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = "{}.pth".format(name)
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, path: str):
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                "No checkpoint found. Initializing model from scratch"
            )
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))

        checkpoint = self._load_file(path)
        self._load_model(checkpoint)
        for key, obj in self.checkpointables.items():
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)

    def resume_or_load(self, path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint():
            path = self.get_checkpoint_file()
        return self.load(path)

    def tag_last_checkpoint(self, last_filename_basename: str):
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str):
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.
        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint: Any):
        """
        Load weights from a checkpoint.
        Args:
            checkpoint (Any): checkpoint contains the weights.
        """
        checkpoint_state_dict = checkpoint.pop("model")

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        model_state_dict = self.model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    self.logger.warning(
                        "'{}' has shape {} in the checkpoint but {} in the "
                        "model! Skipped.".format(
                            k, shape_checkpoint, shape_model
                        )
                    )
                    checkpoint_state_dict.pop(k)

        incompatible = self.model.load_state_dict(
            checkpoint_state_dict, strict=False
        )
        if incompatible.missing_keys:
            self.logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            self.logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, args, *, save_to_disk=True, **checkpointables):
        super().__init__(
            model,
            args.save_folder,
            save_to_disk=save_to_disk,
            **checkpointables,
        )
        self.phase = args.phase

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    # def _load_model(self, checkpoint):
    #     if self.cfg is not None and self.cfg.MODEL.ROI_HEADS.NAME == 'ReweightedROIHeads_Incre':
    #         logger = logging.getLogger(__name__)
    #         logger.info("Initializing box_head for noval classes.")
    #         dict = {}
    #         for key, value in checkpoint["model"].items():
    #             if 'box_head' in key:
    #                 key = key.replace('box_head', 'box_head_noval')
    #                 dict[key] = value
    #         checkpoint["model"].update(dict)
    #     # for non-caffe2 models, use standard ways to load it
    #     super()._load_model(checkpoint)

    def load(self, path: str):
        """
        Load from the given checkpoint. When path points to network file, this
        function has to be called on all ranks.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info(
                "No checkpoint found. Initializing model from scratch"
            )
            return {}
        self.logger.info("Loading checkpoint from {}".format(path))

        checkpoint = self._load_file(path)
        if 'vgg16_reducedfc' in path:
            checkpoint['model'] = OrderedDict({'base.'+key: value for key, value in checkpoint['model'].items()})
        self._load_model(checkpoint)
        if self.phase == 2:
            self.checkpointables = {}
            try:
                checkpoint.pop('iteration')
            except KeyError:
                pass
        for key, obj in self.checkpointables.items():
            if key in checkpoint:
                self.logger.info("Loading {} from {}".format(key, path))
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint


class PeriodicCheckpointer:
    """
    Save checkpoints periodically. When `.step(iteration)` is called, it will
    execute `checkpointer.save` on the given checkpointer, if iteration is a
    multiple of period or if `max_iter` is reached.
    """

    def __init__(self, checkpointer: Any, period: int, max_iter: int = None):
        """
        Args:
            checkpointer (Any): the checkpointer object used to save
            checkpoints.
            period (int): the period to save checkpoint.
            max_iter (int): maximum number of iterations. When it is reached,
                a checkpoint named "model_final" will be saved.
        """
        self.checkpointer = checkpointer
        self.period = int(period)
        self.max_iter = max_iter

    def step(self, iteration: int, **kwargs: Any):
        """
        Perform the appropriate action at the given iteration.

        Args:
            iteration (int): the current iteration, ranged in [0, max_iter-1].
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)
        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(
                "model_{:07d}".format(iteration), **additional_state
            )
        if iteration >= self.max_iter - 1:
            self.checkpointer.save("model_final", **additional_state)

    def save(self, name: str, **kwargs: Any):
        """
        Same argument as :meth:`Checkpointer.save`.
        Use this method to manually save checkpoints outside the schedule.

        Args:
            name (str): file name.
            kwargs (Any): extra data to save, same as in
                :meth:`Checkpointer.save`.
        """
        self.checkpointer.save(name, **kwargs)


def get_missing_parameters_message(keys: list):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the model but not found in a checkpoint.
    Args:
        keys (list[str]): List of keys that were not found in the checkpoint.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "Some model parameters are not in the checkpoint:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "blue") for k, v in groups.items()
    )
    return msg


def get_unexpected_parameters_message(keys: list):
    """
    Get a logging-friendly message to report parameter names (keys) that are in
    the checkpoint but not found in the model.
    Args:
        keys (list[str]): List of keys that were not found in the model.
    Returns:
        str: message.
    """
    groups = _group_checkpoint_keys(keys)
    msg = "The checkpoint contains parameters not used by the model:\n"
    msg += "\n".join(
        "  " + colored(k + _group_to_str(v), "magenta")
        for k, v in groups.items()
    )
    return msg


def _strip_prefix_if_present(state_dict: collections.OrderedDict, prefix: str):
    """
    Strip the prefix in metadata, if any.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
        return

    for key in keys:
        newkey = key[len(prefix) :]
        state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata, if any..
    try:
        metadata = state_dict._metadata
    except AttributeError:
        pass
    else:
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


def _group_checkpoint_keys(keys: list):
    """
    Group keys based on common prefixes. A prefix is the string up to the final
    "." in each key.
    Args:
        keys (list[str]): list of parameter names, i.e. keys in the model
            checkpoint dict.
    Returns:
        dict[list]: keys with common prefixes are grouped into lists.
    """
    groups = defaultdict(list)
    for key in keys:
        pos = key.rfind(".")
        if pos >= 0:
            head, tail = key[:pos], [key[pos + 1 :]]
        else:
            head, tail = key, []
        groups[head].extend(tail)
    return groups


def _group_to_str(group: list):
    """
    Format a group of parameter name suffixes into a loggable string.
    Args:
        group (list[str]): list of parameter name suffixes.
    Returns:
        str: formated string.
    """
    if len(group) == 0:
        return ""

    if len(group) == 1:
        return "." + group[0]

    return ".{" + ", ".join(group) + "}"
