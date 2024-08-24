
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.3.0+cu121
# torch cuda version: 12.1
# torch git version: 97ff6cfd9c86c5c09d7ce775ab64ec5c99230f5d


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 

# GPU Hardware Info: 
# NVIDIA H100 80GB HBM3 : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        neg = torch.ops.aten.neg.default(arg0_1);  arg0_1 = None
        maximum = torch.ops.aten.maximum.default(arg1_1, neg);  arg1_1 = neg = None
        clone = torch.ops.aten.clone.default(maximum);  maximum = None
        return (clone,)
        
def load_args(reader):
    buf0 = reader.storage(None, 8, dtype_hint=torch.uint8)
    reader.tensor(buf0, (1, 8), dtype=torch.uint8, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 67120, dtype_hint=torch.uint8)
    reader.tensor(buf1, (8390, 8), dtype=torch.uint8, is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
