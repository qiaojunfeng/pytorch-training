# Understanding the effect of ZeRo-{1, 2, 3} on memory

Here we are going to check the memory consumption of the different stages of ZeRo.
This is based on PyTorch's example [Shard optimizer states with `ZeroRedundancyOptimizer`](https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html).

In the script [`deepspeed_check_mem.py`](deepspeed_check_mem.py) we define a simple model with different number of parameters and measure the total allocated memory on the GPU at different points of the execution.

```bash
ssh class424@ela.cscs.ch
ssh class424@daint.cscs.ch

salloc -N 8 -Cgpu -Aclass08 --res pytorch_training
. /apps/daint/UES/6.0.UP04/sandboxes/sarafael/hpcpython2022/bin/activate
```

```bash
srun --pty python deepspeed_check_mem.py --deepspeed_config ds_config.json --data-dim 10000
```
