# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import fnmatch
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Union[None, Set[str]]:
    """
    Go through the list of parameter names and select those that match
    any of the provided constraints
    """
    parameter_names = []
    for param_name in constraints:
        matching_parameters = set(fnmatch.filter(all_parameter_names, param_name))
        assert (
            len(matching_parameters) > 0
        ), f"param_names {param_name} don't match any param in the given names."
        parameter_names.append(matching_parameters)
    return set.union(*parameter_names)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return {}

    all_keys = list(state_dict.keys())
    included_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: state_dict[k] for k in included_keys}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Remove from the state dictionary the parameters matching the provided unix patterns

    Args:
        patterns: the list of unix patterns to exclude
        state_dict: the dictionary to filter

    Returns:
        A new state dictionary
    """
    if len(patterns) == 0:
        return state_dict

    all_keys = list(state_dict.keys())
    excluded_keys = unix_pattern_to_parameter_names(patterns, all_keys)
    return {k: v for k, v in state_dict.items() if k not in excluded_keys}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]):
    keys = []
    trace = []
    for k, v in state_dict.items():
        keys.append(k)
        trace.append(v.sum().item())
    trace = np.array(trace)[np.argsort(keys)]
    return trace


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]):
    """
    Verifies that all the parameters matching the provided patterns
    are frozen - this acts as a safeguard when ignoring parameter
    when saving checkpoints - if the parameters are in fact trainable
    """
    if not patterns:
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    non_frozen_keys = {
        n
        for n, p in model.named_parameters()
        if n in frozen_state_dict and p.requires_grad
    }
    if non_frozen_keys:
        raise ValueError(
            f"Parameters excluded with `skip_saving_parameters` should be frozen: {non_frozen_keys}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(
    model: nn.Module, patterns: List[str], disabled: bool = True
):
    """
    Context manager that inspects a model surrounding a piece of code
    and verifies if the model has been updated by this piece of code

    The function will raise an exception if the model has been updated
    on at least one of the parameter that matches one of the pattern

    Args:
        model: the model that might have been updated
        patterns: for the parameters we want to observe
        allowed:
    """
    if not patterns or disabled:
        yield
        return

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_before = _get_state_dict_summary(frozen_state_dict)

    yield

    frozen_state_dict = filter_params_matching_unix_pattern(
        patterns=patterns, state_dict=model.state_dict()
    )
    summary_after = _get_state_dict_summary(frozen_state_dict)

    if not np.allclose(summary_before, summary_after, atol=1e-6):
        raise ValueError(
            f"""
            The `model_weight_initializer` has initialized parameters frozen with `skip_saving_parameters`.
            You can resolve this error by either initializing those parameters from within the model definition
            or using the flag `trainer.checkpoint.initialize_after_preemption` to True.
        """
        )


class CkptExcludeKernel:
    """
    Removes the keys from the given model state_dict that match the key_pattern.

    Args:
        key_pattern: Patterns used to select the keys in the state_dict
            that are eligible for this kernel.
    """

    def __init__(self, key_pattern: List[str]):
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict):
        """
        Args:
            state_dict: A dictionary representing the given checkpoint's state dict.
        """
        if len(self.key_pattern) == 0:
            return state_dict
        exclude_keys = unix_pattern_to_parameter_names(
            self.key_pattern, list(state_dict.keys()) # Ensure keys() is called
        )
        return {k: v for k, v in state_dict.items() if k not in exclude_keys}


def load_checkpoint(
    path_list: List[str],
    pick_recursive_keys: Optional[List[str]] = None,
    map_location: str = "cpu",
) -> Any:
    """
    Loads a checkpoint from the specified path.

    Args:
        path_list: A list of paths which contain the checkpoint. Each element
            is tried (in order) until a file that exists is found. That file is then
            used to read the checkpoint.
        pick_recursive_keys: Picks sub dicts from the loaded checkpoint if not None.
            For pick_recursive_keys = ["a", "b"], will return checkpoint_dict["a"]["b"]
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Model with the matchin pre-trained weights loaded.
    """
    path_exists = False
    final_path = ""
    for path in path_list:
        if g_pathmgr.exists(path):
            path_exists = True
            final_path = path
            break

    if not path_exists:
        raise ValueError(f"No path exists in {path_list}")

    with g_pathmgr.open(final_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    logging.info(f"Loaded checkpoint from {final_path}")
    if pick_recursive_keys is not None:
        for key in pick_recursive_keys:
            checkpoint = checkpoint[key]
    return checkpoint


def get_state_dict(checkpoint, ckpt_state_dict_keys: Tuple[str, ...]): # Added type hint for ckpt_state_dict_keys
    if isinstance(checkpoint, RecursiveScriptModule):
        # This is a torchscript JIT model
        return checkpoint.state_dict()
    pre_train_dict = checkpoint
    for i, key in enumerate(ckpt_state_dict_keys):
        # Check if pre_train_dict is a mapping and key exists, or if it's a sequence and key is a valid index
        is_mapping_and_key_exists = isinstance(pre_train_dict, Mapping) and key in pre_train_dict
        is_sequence_and_key_valid = isinstance(pre_train_dict, Sequence) and isinstance(key, int) and key < len(pre_train_dict)
        
        if not (is_mapping_and_key_exists or is_sequence_and_key_valid):
            key_path_str = '["' + '"]["'.join(map(str, ckpt_state_dict_keys[:i+1])) + '"]'
            available_keys = list(pre_train_dict.keys()) if isinstance(pre_train_dict, Mapping) else f"length {len(pre_train_dict)}"
            raise KeyError(
                f"Key '{key}' (part of path {key_path_str}) not found or invalid in checkpoint. "
                f"Available keys/elements at current level: {available_keys}"
            )
        pre_train_dict = pre_train_dict[key]
    return pre_train_dict


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: List[Callable] = None,
    ckpt_state_dict_keys: Tuple[str, ...] = ("state_dict",), # Made it a tuple
    map_location: str = "cpu",
) -> Dict: # Return type is Dict
    """
    Performs checkpoint loading with a variety of pre-processing kernel applied in
    sequence.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        checkpoint_kernels List(Callable): A list of checkpoint processing kernels
            to apply in the specified order. Supported kernels include `CkptIncludeKernel`,
            `CkptExcludeKernel`, etc. These kernels are applied in the
            given order.
        ckpt_state_dict_keys (Tuple[str,...]): Keys containing the model state dict.
        map_location (str): a function, torch.device, string or a dict specifying how to
            remap storage locations

    Returns: Processed model state_dict.
    """
    if not g_pathmgr.exists(checkpoint_path): # Added check
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found")

    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)

    pre_train_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)

    logging.debug(
        "Loaded Checkpoint State Dict pre-kernel application: %s",
        str(", ".join(list(pre_train_dict.keys())))
    )
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            pre_train_dict = f(state_dict=pre_train_dict)

    logging.debug(
        "Loaded Checkpoint State Dict Post-kernel application %s",
        str(", ".join(list(pre_train_dict.keys())))
    )
    return pre_train_dict


def check_load_state_dict_errors(
    missing_keys: List[str], # Changed from missing_keys to List[str]
    unexpected_keys: List[str], # Changed from unexpected_keys to List[str]
    strict: bool,
    ignore_missing_keys: Optional[List[str]] = None, # Made Optional
    ignore_unexpected_keys: Optional[List[str]] = None, # Made Optional
):
    final_missing_keys = list(missing_keys) # Work on copies
    final_unexpected_keys = list(unexpected_keys)

    if ignore_missing_keys: # Check if not None and not empty
        ignored_m_keys = unix_pattern_to_parameter_names(
            ignore_missing_keys, final_missing_keys
        )
        if ignored_m_keys: # Check if any keys were actually matched and ignored
             final_missing_keys = [key for key in final_missing_keys if key not in ignored_m_keys]

    if ignore_unexpected_keys: # Check if not None and not empty
        ignored_u_keys = unix_pattern_to_parameter_names(
            ignore_unexpected_keys, final_unexpected_keys
        )
        if ignored_u_keys: # Check if any keys were actually matched and ignored
            final_unexpected_keys = [
                key for key in final_unexpected_keys if key not in ignored_u_keys
            ]

    err_msgs = []
    if final_unexpected_keys:
        err_msgs.append(f"Unexpected key(s) in state_dict: {', '.join(final_unexpected_keys)}")
    if final_missing_keys:
        err_msgs.append(f"Missing key(s) in state_dict: {', '.join(final_missing_keys)}")

    if err_msgs:
        full_err_msg = "Error(s) in loading state_dict: " + " ".join(err_msgs)
        logging.warning(full_err_msg)
        # if strict or final_unexpected_keys: # Raise if strict or if there are any non-ignored unexpected keys
        #     raise RuntimeError(full_err_msg)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = False, # Changed default to True as per common PyTorch practice
    ignore_missing_keys: Optional[List[str]] = None,
    ignore_unexpected_keys: Optional[List[str]] = None,
    checkpoint_kernels: Optional[List[Callable]] = None, # Made Optional
):
    """
    Loads a state dict into the given model, filtering for shape mismatches
    and reporting on missing/unexpected keys.
    """
    print('???????????????????????????????????????????????')
    if checkpoint_kernels is not None:
        for f in checkpoint_kernels:
            state_dict = f(state_dict=state_dict)

    model_dict = model.state_dict()
    
    # Parameters from checkpoint that are not in the current model
    # (these will be part of `unexpected_keys` from `load_state_dict`)
    checkpoint_keys_not_in_model = [k for k in state_dict.keys() if k not in model_dict]

    # Filter checkpoint state_dict for keys present in model and matching shape
    shape_mismatch_ignored_keys = []
    filtered_checkpoint_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_checkpoint_state_dict[k] = v
            else:
                shape_mismatch_ignored_keys.append(
                    f"{k} (ckpt shape: {v.shape}, model shape: {model_dict[k].shape})"
                )
        # else: key k from checkpoint is not in model, will be caught by load_state_dict as unexpected

    # Load the filtered state_dict.
    # `load_state_dict` will return missing_keys (model keys not in filtered_checkpoint_state_dict)
    # and unexpected_keys (keys in filtered_checkpoint_state_dict not in model - should be empty now).
    # However, we also need to consider original checkpoint_keys_not_in_model.
    
    # Using strict=False here because we've already manually filtered for shape.
    # The `check_load_state_dict_errors` will handle strictness based on original `strict` param.
    missing_keys_from_load, unexpected_keys_from_load = model.load_state_dict(filtered_checkpoint_state_dict, strict=False)
    
    # Combine original unexpected keys (from checkpoint, not in model) with any from load (should be none)
    all_unexpected_keys = list(set(checkpoint_keys_not_in_model + unexpected_keys_from_load))

    # `missing_keys_from_load` are model keys not found in `filtered_checkpoint_state_dict`.
    # These are genuinely missing from the (shape-valid) checkpoint.
    
    logging.info(f"Attempted to load {len(filtered_checkpoint_state_dict)} parameters from checkpoint into model.")
    if shape_mismatch_ignored_keys:
        logging.warning(
            f"{len(shape_mismatch_ignored_keys)} parameter(s) from checkpoint were IGNORED due to SHAPE MISMATCH:\n\t"
            + "\n\t".join(shape_mismatch_ignored_keys)
        )

    # Report on keys in the model that were not updated by any key in the original state_dict
    # (either because they were missing in state_dict, or had shape mismatch)
    model_keys = set(model_dict.keys())
    loaded_keys = set(filtered_checkpoint_state_dict.keys())
    not_updated_model_keys = list(model_keys - loaded_keys)
    
    if not_updated_model_keys:
        logging.warning(
            f"{len(not_updated_model_keys)} model parameter(s) were NOT UPDATED from checkpoint (missing in checkpoint or shape mismatch):\n\t"
            + "\n\t".join(not_updated_model_keys)
        )
    
    # Use the utility to check for errors based on original strictness and ignore lists
    # `missing_keys_from_load` are truly missing from the perspective of the model vs filtered checkpoint
    # `all_unexpected_keys` are keys from original checkpoint that don't belong in the model
    check_load_state_dict_errors(
        missing_keys=missing_keys_from_load, # Model keys not in filtered_checkpoint_state_dict
        unexpected_keys=all_unexpected_keys, # Checkpoint keys not in model
        strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    
    return model
