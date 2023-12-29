# Multiprocess Data Loader for ML Training

## Installation

Install the dependencies in `requirements.txt` then simply run 

```bash
pip install --upgrade git+https://github.com/viebboy/swift-loader.git
```

## Quick Start
Given your `Dataset` class that exposes `__getitem__` and `__len__`, you can create a `SwiftLoader` as follows:

```python3

from swift_loader import SwiftLoader

loader = SwiftLoader(
    dataset_class=Dataset,
    dataset_kwargs=dataset_kwargs, # keyword-arguments for dataset construction,
    batch_size=32, # integer defining batch size
    nb_consumer=nb_gpu, # should be the number of GPUs used for training
    worker_per_consumer=4, # number of child processes that loads & batch data for a consumer (a GPU)
    shuffle=True,
    **kwargs,
)
```

In the above code, `nb_consumer` specifies the number of GPUs that will be used for training or evaluation.   

If you're training using 4 GPUs, set `nb_consumer=4`.  

Each consumer (GPU) will effectively get `1/4` data from the dataset, which is loaded and batched together by `4` child 
processes (`nb_worker_per_consumer`).  

For the best use of resources, `nb_consumer * worker_per_consumer` should not exceed the number of CPU cores in your machine. 

The `batch_size` determines the minibatch size produced for each consumer (each GPU).  

Then before starting to consume `loader` in your code, `loader.start()` should be called once to signal the loader to start loading:

```python3

loader.start(consumer_index: int, device: torch.device = None, to_device: Callable = None)
```

When calling `start()`, the index (starting from 0) of the consumer should be provided so the `loader` knows
which part of dataset to work on.

Additionally, if `device` is specified (default is `None`), minibatch will be moved to the specified device.  

When `device` is **not** `None` and if `to_device` **is** None (not provided), `SwiftLoader` will 
use the default function that recursively moves all `numpy` or `torch.Tensor` to the corresponding device. 


## Examples

If you're familiar with `Fabric` from `Lightning AI`, please checkout [fabric_example.py](./examples/fabric_example.py)


## All Aguments

In addition to the mandatory arguments, the following keyword arguments are supported:

- `seed_numer`: (int) this number is used to in builtin `random` module. Default to `int(time.time()*1000)`
- `data_queue_size`: (int) number of minibatches in the queues (queue in child processes, queue in parent process). Default to `10`
- `message_queue_size`: (int) number of telcom messages (not minibatches) that can stay in pipe. Default to `10`
- `collate_fn`: (callable) the function used to collate/batch samples. If not provided, `SwiftLoader` will use the default function, which only works with nested lists.
- `logger`: (str, dict): if a `str` is provided, it should be the file prefix to save log files. If a `dict` is provided, it could have the following keys:
    - `path`: (str) file prefix to dump the log files
    - `stdout`: (bool) if True, will print to standard output
    - `level`: (str) log level
- `batch_encoder`: (callable) because minibatches are prepared in child processes, they need to be serialized. Users can provide custom function to serialize a minibatch to bytes. If not provided, `dill.dumps` will be used. 
- `batch_decoder`: (callable) if `batch_encoder` is provided, users should also provide `batch_encoder` to reconstruct a minibatch from bytes. Default to `dill.loads`.
- `nearby_shuffle`: (int) this number limits the amount of randomness when getting samples from dataset. 
  Higher numbers lead to more randomness but might also affect performance of getting samples from the dataset if the implementation involves accessing a large binary files. 
  Default to `5 * batch_size`
- `benchmark`: (bool) whether to perform various benchmark of dataloading and so on. Default to `True`. 
- `benchmark_file`: (str) if provided, the benchmark results are written to this file. Otherwise, it will be written to the log file if `logger` is defined. 
- `validate`: (bool) whether to test run the dataset and batching step before launching the workers. Default to `True`. 
  If constructing a dataset is expensive and you are certain that dataset creation and batching have no issue, you can turn off this step to save some time. 
- `multithread`: (bool) if True, will use multiple threads, each thread will handle the telcom with one worker. Default to True. 
- `max_nb_batch_on_device`: (int) if data should be moved to device, this defines how many minibatches will stay on the device at the same time. Default to `1`
- `max_buffer_size`: maximum size of the buffer when reconstructing data from workers. Default to `max(2, data_queue_size//worker_per_consumer)`


## Authors
Dat Tran (hello@dats.bio)
