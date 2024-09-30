import time

import numpy as np
import torch
from tqdm import tqdm


def benchmark_model(call_model, model_args, n_warmups=50, n_tests=500):
    elapsed_times = []

    for iteration in tqdm(range(n_warmups + n_tests)):
        start = time.time()
        call_model(*model_args)
        torch.cuda.synchronize()
        end = time.time()

        if iteration >= n_warmups:
            elapsed_times.append(end - start)

    result = np.mean(elapsed_times)
    print(result)

    return result
