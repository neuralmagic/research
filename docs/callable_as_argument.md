# How to pass a Python callable as an argument to a ClearML task


In most occasions, ClearML tasks created by this library are executed remotely.
As a consequence, arguments passed to the task must be serialized, stored into the ClearML task, and then accessed by the script executing the task remotely.
Arguments are preferrably stored as `arguments` or `configurations`.
However, these interfaces do not easily support python callables.

In order to reliably store a callable into the ClearML task, it is preferrable to store the callable as an artifact and manually handle serialization and deserialization.
ClearML uses `pickle` to serialize Python objects, but experience shows that this is not a reliable method to serialize callable objects in general.
Instead, the suggested pathway is to store the code used in the callable definition as text using `inspect.getgetsource`, and then to use `exec` to re-create the callable in the remote server.

The example below is used for multimodal models in llm-compressor:

**Serialization code:**
```python
# callable definition
def data_collator(x):
    import torch
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

# serialization
task.upload_artifact("data collator", inspect.getsource(data_collator))
```

**Deserialization code:**
```python
filepath = task.artifacts["data collator"].get_local_copy()
namespace = {}
exec(open(filepath, "r").read(), namespace)
data_collator_fn = namespace.get(data_collator_fn_name)
```