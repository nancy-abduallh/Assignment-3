import torch
import torch.utils.data.dataloader
import importlib
import collections
import traceback
import numbers
import numpy as np
from pathlib import Path

# Local utility types (these must exist in your repo)
from lib.utils import TensorDict, TensorList

# Use modern ABCs
from collections.abc import Mapping, Sequence

int_classes = int  # legacy alias kept


def _check_use_shared_memory():
    """Safe check for shared memory flag (backwards compatible)."""
    try:
        if hasattr(torch.utils.data.dataloader, '_use_shared_memory'):
            return getattr(torch.utils.data.dataloader, '_use_shared_memory')
        collate_lib = importlib.import_module('torch.utils.data._utils.collate')
        if hasattr(collate_lib, '_use_shared_memory'):
            return getattr(collate_lib, '_use_shared_memory')
    except Exception:
        return False
    return torch.utils.data.get_worker_info() is not None


def safe_stack(tensors, dim=0):
    """Try torch.stack; if shapes mismatch slightly trim to smallest along leading dims."""
    try:
        return torch.stack(tensors, dim=dim)
    except RuntimeError:
        # Fallback: trim every tensor to the minimal shape across batch (only works for >=1D tensors)
        try:
            shapes = [t.shape for t in tensors]
            if len(shapes) == 0 or not hasattr(shapes[0], '__len__'):
                # can't fix, raise original
                raise
            min_shape = [min(s[i] if i < len(s) else 0 for s in shapes) for i in range(len(shapes[0]))]
            # slice tensors to min_shape on each dimension
            new_tensors = []
            for t in tensors:
                slices = tuple(slice(0, min_shape[d]) for d in range(len(min_shape)))
                new_tensors.append(t[slices])
            return torch.stack(new_tensors, dim=dim)
        except Exception:
            # re-raise original if trimming fails
            raise


def _is_numpy_array(obj):
    return isinstance(obj, np.ndarray)


def _numpy_to_tensor(arr):
    # Convert common numpy dtypes to torch tensors
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    return torch.from_numpy(arr)


def _all_same_signature(batch):
    """
    Return True if all elements in batch share the same "type signature".
    Prevents infinite recursion when elements are heterogeneous.
    """
    if len(batch) == 0:
        return True
    first = batch[0]
    first_type = type(first)
    # For numpy arrays and tensors consider dtype/ndim as part of signature
    if isinstance(first, torch.Tensor):
        sig = (torch.Tensor, first.dim())
    elif _is_numpy_array(first):
        sig = (np.ndarray, first.ndim)
    else:
        sig = first_type
    for elem in batch[1:]:
        if isinstance(elem, torch.Tensor):
            elem_sig = (torch.Tensor, elem.dim())
        elif _is_numpy_array(elem):
            elem_sig = (np.ndarray, elem.ndim)
        else:
            elem_sig = type(elem)
        if elem_sig != sig:
            return False
    return True


def ltr_collate(batch):
    """
    Collate function: similar to PyTorch default but supports:
     - lib.utils.TensorDict and TensorList
     - numpy arrays -> torch tensors
     - safe stacking for slightly mismatched tensors
    """
    if len(batch) == 0:
        return None

    elem = batch[0]

    # Torch tensor
    if isinstance(elem, torch.Tensor):
        return safe_stack(batch, 0)

    # numpy arrays
    if _is_numpy_array(elem):
        try:
            tensors = [_numpy_to_tensor(b) for b in batch]
            return safe_stack(tensors, 0)
        except Exception:
            return batch

    # numbers
    if isinstance(elem, numbers.Number):
        # choose appropriate tensor type
        if isinstance(elem, int):
            return torch.LongTensor(batch)
        else:
            return torch.DoubleTensor(batch)

    # strings / bytes -> return list as-is (we don't stack strings)
    if isinstance(elem, (str, bytes)):
        return batch

    # TensorDict (assumed mapping-like object)
    if isinstance(elem, TensorDict):
        return TensorDict({k: ltr_collate([d[k] for d in batch]) for k in elem})

    # TensorList (assumed sequence-of-items container)
    if isinstance(elem, TensorList):
        # transpose list-of-lists and collate per slot
        transposed = list(zip(*batch))
        return TensorList([ltr_collate(list(samples)) for samples in transposed])

    # Mapping/dict
    if isinstance(elem, Mapping):
        return {key: ltr_collate([d[key] for d in batch]) for key in elem}

    # Sequence (but not string/bytes) -> transpose and recurse only if homogeneous
    if isinstance(elem, Sequence):
        # prevent infinite recursion on very heterogeneous nested sequences
        if not _all_same_signature(batch):
            # fall back to python list to avoid deep recursion
            return list(batch)
        transposed = list(zip(*batch))
        return [ltr_collate(list(samples)) for samples in transposed]

    # fallback: unknown type -> return list
    return list(batch)


def ltr_collate_stack1(batch):
    """
    Same as ltr_collate but stacks tensors along dim=1 when appropriate.
    """
    if len(batch) == 0:
        return None

    elem = batch[0]

    if isinstance(elem, torch.Tensor):
        return safe_stack(batch, 1)

    if _is_numpy_array(elem):
        try:
            tensors = [_numpy_to_tensor(b) for b in batch]
            return safe_stack(tensors, 1)
        except Exception:
            return batch

    if isinstance(elem, numbers.Number):
        if isinstance(elem, int):
            return torch.LongTensor(batch)
        else:
            return torch.DoubleTensor(batch)

    if isinstance(elem, (str, bytes)):
        return batch

    if isinstance(elem, TensorDict):
        return TensorDict({k: ltr_collate_stack1([d[k] for d in batch]) for k in elem})

    if isinstance(elem, TensorList):
        transposed = list(zip(*batch))
        return TensorList([ltr_collate_stack1(list(samples)) for samples in transposed])

    if isinstance(elem, Mapping):
        return {key: ltr_collate_stack1([d[key] for d in batch]) for key in elem}

    if isinstance(elem, Sequence):
        if not _all_same_signature(batch):
            return list(batch)
        transposed = list(zip(*batch))
        return [ltr_collate_stack1(list(samples)) for samples in transposed]

    return list(batch)


class LTRLoader(torch.utils.data.dataloader.DataLoader):
    """
    Safe, memory-robust DataLoader for SeqTrack with a robust collate function.

    Notes:
    - Bounds num_workers to reasonable defaults to avoid worker crashes on low-RAM machines.
    - Disables pin_memory on small GPUs automatically.
    - Attempts to restart iteration if workers crash (useful on OOM/worker-killed cases).
    """

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, persistent_workers=False):
        # Choose appropriate collate
        if collate_fn is None:
            collate_fn = ltr_collate_stack1 if stack_dim == 1 else ltr_collate

        # Safety: limit num_workers and pin_memory on small devices
        num_workers = int(max(0, min(num_workers, 4)))
        try:
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory
                if total_mem < 8 * 1024 ** 3:  # < 8GB
                    pin_memory = False
        except Exception:
            pin_memory = False

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers
        )

        self.name = name
        self.training = training
        self.stack_dim = stack_dim
        self.epoch_interval = epoch_interval

    def __iter__(self):
        """
        Wrap iteration with worker-crash handling.
        If a worker is killed (OOM or killed), try to clear cache and restart iteration once.
        """
        try:
            for batch in super().__iter__():
                yield batch
        except RuntimeError as e:
            print(f"⚠️ [LTRLoader] Worker crashed during iteration: {e}. Attempting to recover...")
            traceback.print_exc()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Try one more time (re-create iterator)
            try:
                for batch in super().__iter__():
                    yield batch
            except Exception as e2:
                print("⚠️ [LTRLoader] Recovery attempt failed; re-raising exception.")
                raise e2
