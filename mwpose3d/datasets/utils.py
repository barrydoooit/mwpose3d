from typing import Any, List, Mapping, Sequence


def pseudo_collate(data_batch: Sequence) -> Any:
    data_item = data_batch[0]
    data_item_type = type(data_item)
    if isinstance(data_item, (str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named tuple
        return data_item_type(*(pseudo_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [pseudo_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [pseudo_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [pseudo_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        return data_item_type({
            key: pseudo_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return data_batch
    

# -----------Decollate utils------------
def _is_container(x):
    return isinstance(x, Mapping) or (isinstance(x, Sequence) and not isinstance(x, (str, bytes)))

def _infer_batch_size(batched_obj) -> int:
    if isinstance(batched_obj, Mapping):
        return _infer_batch_size(next(iter(batched_obj.values())))
    if isinstance(batched_obj, Sequence) and not isinstance(batched_obj, (str, bytes)):
        if len(batched_obj) == 0:
            return 0
        # If elements are NOT containers, this level is already [items per sample]
        if not any(_is_container(el) for el in batched_obj):
            return len(batched_obj)
        # Otherwise keep descending
        return _infer_batch_size(batched_obj[0])
    # Shouldn't reach here for well-formed batched data
    raise TypeError("Cannot infer batch size from object of type {}".format(type(batched_obj)))

def pseudo_decollate(batched: Any, batch_size: int | None = None) -> List[Any]:
    if batch_size is None:
        batch_size = _infer_batch_size(batched)

    if isinstance(batched, Mapping):
        # decollate each value -> list of length N, then zip into dicts
        dec = {k: pseudo_decollate(v, batch_size) for k, v in batched.items()}
        return [{k: dec[k][i] for k in dec.keys()} for i in range(batch_size)]

    if isinstance(batched, tuple) and hasattr(batched, "_fields"):
        # namedtuple of batched fields
        fields_lists = [pseudo_decollate(x, batch_size) for x in batched]
        return [type(batched)(*(fl[i] for fl in fields_lists)) for i in range(batch_size)]

    if isinstance(batched, Sequence) and not isinstance(batched, (str, bytes)):
        if len(batched) == 0:
            return [[] for _ in range(batch_size)]
        # Leaf case: list of per-sample scalars/tensors/objects
        if not any(_is_container(el) for el in batched) and len(batched) == batch_size:
            return list(batched)
        # Structured sequence: each element is a batched field; decollate each then reassemble per sample
        elems_lists = [pseudo_decollate(el, batch_size) for el in batched]
        out = []
        for i in range(batch_size):
            try:
                out.append(type(batched)(e[i] for e in elems_lists))
            except TypeError:
                out.append(tuple(e[i] for e in elems_lists))
        return out

    # Reaching here means we have a scalar/tensor, but the batched leaf should have been a list already.
    # Treat as replicated value across batch.
    return [batched for _ in range(batch_size)]

def pseudo_recollate_into(samples: List[Any], out: Any) -> Any:
    """
    Mutate `out` so it holds the collated view of `samples`, reusing containers
    when they are mutable (dict/list). Returns `out` (which may be replaced at
    immutable levels).
    Assumes `samples` is a non-empty list of per-sample items that would be
    valid input to `pseudo_collate`.
    """
    s0 = samples[0]

    # Mapping: keep top-level dict identity; recurse per key
    if isinstance(s0, Mapping):
        # Optionally drop keys not present anymore
        for k in list(out.keys()):
            if k not in s0:
                out.pop(k)
        for k in s0:
            vals_k = [s[k] for s in samples]
            if k in out:
                out[k] = pseudo_recollate_into(vals_k, out[k])
            else:
                out[k] = pseudo_collate(vals_k)  # first-time allocation
        return out

    # Sequence (but not str/bytes)
    if isinstance(s0, Sequence) and not isinstance(s0, (str, bytes)):
        # group per field across batch (like your collate)
        transposed = list(zip(*samples))  # small, predictable allocation

        # If `out` is a list (mutable), resize and fill in place
        if isinstance(out, list):
            if len(out) != len(transposed):
                out[:] = [None] * len(transposed)
            for i, group in enumerate(transposed):
                if out[i] is None:
                    out[i] = pseudo_collate(list(group))
                else:
                    out[i] = pseudo_recollate_into(list(group), out[i])
            return out

        # Immutable or custom sequence: can’t mutate -> rebuild and return
        return type(out)(pseudo_recollate_into(list(group), elem) if i < len(out) else pseudo_collate(list(group))
                         for i, (group, elem) in enumerate(zip(transposed, out)))

    # Leaf level: `out` should be the batched leaf (usually a list)
    if isinstance(out, list):
        # overwrite in place to preserve list identity
        if len(out) != len(samples):
            out[:] = [None] * len(samples)
        out[:] = samples
        return out

    # If we get a non-list leaf container, we can’t mutate; allocate new
    return list(samples)