import ctypes
import os
import sys
import torch
from typing import Callable, Iterable, Optional, Union


def _move_tensors(obj, device: Union[str, torch.device], record: dict):
    """Recursively move all Tensors in `obj` to `device`, recording original devices."""
    if isinstance(obj, torch.Tensor):
        moved = obj.to(device)
        record[id(moved)] = obj.device
        return moved
    if isinstance(obj, tuple):
        return tuple(_move_tensors(x, device, record) for x in obj)
    if isinstance(obj, list):
        return [_move_tensors(x, device, record) for x in obj]
    if isinstance(obj, dict):
        return {k: _move_tensors(v, device, record) for k, v in obj.items()}
    return obj


def _restore_tensors(obj, record: dict):
    """Move Tensors back to their original devices using the record from `_move_tensors`."""
    if isinstance(obj, torch.Tensor):
        original_device = record.get(id(obj))
        return obj.to(original_device) if original_device is not None else obj
    if isinstance(obj, tuple):
        return tuple(_restore_tensors(x, record) for x in obj)
    if isinstance(obj, list):
        return [_restore_tensors(x, record) for x in obj]
    if isinstance(obj, dict):
        return {k: _restore_tensors(v, record) for k, v in obj.items()}
    return obj


def _set_cell(cell, value):
    """Set the contents of a closure cell (CPython-specific)."""
    ctypes.pythonapi.PyCell_Set(ctypes.py_object(cell), ctypes.py_object(value))


class _CudaClosureContext:
    """Temporarily move specified closure variables of a function to CUDA and back."""
    def __init__(self, fn: Callable, tensor_vars: Optional[Iterable[str]], device: Union[str, torch.device] = 'cuda'):
        self.fn = fn
        self.tensor_vars = tensor_vars
        self.device = device
        self.cell_info = []

    def __enter__(self):
        if self.tensor_vars is None:
            return self
        free_vars = self.fn.__code__.co_freevars
        closure = self.fn.__closure__ or ()
        name_to_cell = dict(zip(free_vars, closure))
        for name in self.tensor_vars:
            cell = name_to_cell.get(name)
            if cell is None:
                continue
            record = {}
            moved = _move_tensors(cell.cell_contents, self.device, record)
            _set_cell(cell, moved)
            self.cell_info.append((cell, record))
        return self

    def __exit__(self, *args):
        for cell, record in self.cell_info:
            current = cell.cell_contents
            restored = _restore_tensors(current, record)
            _set_cell(cell, restored)


def bench(fn, num_warmups: int = 5, num_tests: int = 10,
          high_precision: bool = False):
    # Flush L2 cache with 256 MB data
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')
    cache.zero_()

    # Warmup
    for _ in range(num_warmups):
        fn()

    # Add a large kernel to eliminate the CPU launch overhead
    if high_precision:
        x = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        y = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        x @ y

    # Testing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(num_tests):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / num_tests / 1e3


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(fn, kernel_names, num_tests: int = 1,
                 suppress_kineto_output: bool = False,
                 trace_path: str = None, flush_l2: bool = True,
                 with_multiple_kernels: bool = False,
                 barrier: Optional[Callable] = None,
                 tensor_vars: Optional[Iterable[str]] = None):
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)

    # Skip profiling
    # Conflict with Nsight Systems, Nsight Compute and Compute Sanitizer
    if int(os.environ.get('DG_USE_NVIDIA_TOOLS', 0)):
        return (1, ) * len(kernel_names) if is_tuple else 1

    # By default, flush L2 with an excessive 8 GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    with _CudaClosureContext(fn, tensor_vars):
        # For some auto-tuning kernels with prints
        fn()

        # Profile
        suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
        with suppress():
            schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule, acc_events=True)
            with profiler:
                for i in range(2):
                    for _ in range(num_tests):
                        if flush_l2:
                            torch.empty(flush_l2_size, dtype=torch.int, device='cuda').zero_()
                        if barrier is not None:
                            # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                            # noinspection PyProtectedMember
                            torch.cuda._sleep(int(2e7))  # ~10ms
                            barrier()
                        fn()
                    torch.cuda.synchronize()
                    profiler.step()

        # Parse the profiling table
        prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', max_name_column_width=100).split('\n')

    kernel_names = (kernel_names, ) if isinstance(kernel_names, str) else kernel_names
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) <= 1, f'Errors of the kernel {name} in the profiling table {prof_lines}'

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {'ms': 1e3, 'us': 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time = 0
        total_num = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += float(time_str.replace(unit, '')) / scale * int(num_str)
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num if total_num > 0 else 0)

    return tuple(kernel_times) if is_tuple else kernel_times[0]
