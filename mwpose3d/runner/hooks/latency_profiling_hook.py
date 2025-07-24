import json
from pathlib import Path
import time
from typing import Any, List, TYPE_CHECKING, Optional
from mmengine.hooks import Hook
from mwpose3d.registry import HOOKS
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mwpose3d.runner import Runner


@HOOKS.register_module()
class LatencyProfilingHook(Hook):
    def __init__(self, 
                 subject_modules: List[str] = [],
                 on: bool = False,
                 include_full_forward=True,
                 sample_num: Optional[int] = None,
                 out_file: Optional[str] = None):
        self.subject_modules = subject_modules
        self.include_full_forward = include_full_forward # profile also the complete forward pass of the model
        self.out_file = Path(out_file) if out_file else None
        self.on = on 
        self.sample_num = sample_num
        # ---Containers---
        self.records = {} # {module_name: {'cpu': [...], 'gpu': [...]}}
        self._current = {} # temporary storage for the current module
        self.handles = [] # handles for the hooks
        self.summary = {} # summary of the latency for each module
    
    def prepare(self, model: nn.Module):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        self.records.clear()
        self._current.clear()

        if self.include_full_forward:
            pre = model.register_forward_pre_hook(
                lambda m, i: self.on_module_start(m, i, 'full_forward'))
            post = model.register_forward_hook(
                lambda m, i, o: self.on_module_end(m, i, o, 'full_forward'))
            self.handles.extend([pre, post])
        
        for name, module in model.named_modules():
            if (name not in self.subject_modules):
                continue
            else:
                print(name in self.subject_modules)
            print(f'Profiling {name}', self.subject_modules)
            pre = module.register_forward_pre_hook(
                lambda m, i: self.on_module_start(m, i, name))
            post = module.register_forward_hook(
                lambda m, i, o: self.on_module_end(m, i, o, name))
            self.handles.extend([pre, post])
    
    def on_module_start(self, 
                        module: nn.Module, 
                        input: Any,
                        name: str):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        cpu_start = time.perf_counter()

        self._current[name] = (cpu_start, start_event, end_event)
        
    def on_module_end(self, 
                      module: nn.Module, 
                      input: Any, 
                      output: Any,
                      name: str):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        cpu_start, start_event, end_event = self._current.pop(name)
        if cpu_start is None:
            return
        cpu_elapsed = time.perf_counter() - cpu_start
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize()
            gpu_elapsed = start_event.elapsed_time(end_event)
        else:
            gpu_elapsed = None
        
        rec = self.records.setdefault(name, {'cpu': [], 'gpu': []})
        rec['cpu'].append(cpu_elapsed * 1000.0)
        rec['gpu'].append(gpu_elapsed  if gpu_elapsed is not None else None)
    
    def analyze(self):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        self.summary.clear()
        for name, rec in self.records.items():
            count = len(rec['cpu'])
            total_cpu = sum(rec['cpu'])
            total_gpu = sum(rec['gpu']) if rec['gpu'] else None
            self.summary[name] = {
                'profiled_samples': count,
                'cpu': {
                    'mean': total_cpu / count,
                    'max': max(rec['cpu']),
                    'min': min(rec['cpu']),
                    'std': (sum((x - total_cpu / count) ** 2 for x in rec['cpu']) / count) ** 0.5,
                    'median': sorted(rec['cpu'])[count // 2],
                },
                'gpu': {
                    'mean': total_gpu / count,
                    'max': max(rec['gpu']),
                    'min': min(rec['gpu']),
                    'std': (sum((x - total_gpu / count) ** 2 for x in rec['gpu']) / count) ** 0.5,
                    'median': sorted(rec['gpu'])[count // 2],
                } if rec['gpu'] else {},
            }
    
    def dump(self):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return

        if self.out_file:
            self.out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out_file, 'w') as f:
                json.dump(self.summary, f, indent=4)
        
        headers = [
            'Module', 'Samples',
            'CPU Mean (ms)', 'CPU Min', 'CPU Max', 'CPU Std', 'CPU Median',
            'GPU Mean (ms)', 'GPU Min', 'GPU Max', 'GPU Std', 'GPU Median'
        ]
        rows = []
        for name, stats in self.summary.items():
            cpu = stats['cpu']
            gpu = stats.get('gpu', {})
            rows.append([
                name,
                stats['profiled_samples'],
                f"{cpu['mean']:.3f}",
                f"{cpu['min']:.3f}",
                f"{cpu['max']:.3f}",
                f"{cpu['std']:.3f}",
                f"{cpu['median']:.3f}",
                f"{gpu.get('mean', 0.0):.3f}",
                f"{gpu.get('min', 0.0):.3f}",
                f"{gpu.get('max', 0.0):.3f}",
                f"{gpu.get('std', 0.0):.3f}",
                f"{gpu.get('median', 0.0):.3f}",
            ])
        try:
            from tabulate import tabulate
            print(tabulate(rows, headers=headers, tablefmt='github'))
        except ImportError:
            print(' | '.join(headers))
            for r in rows:
                print(' | '.join(map(str, r)))
    
    def before_test_epoch(self, runner: "Runner"):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        model = runner.model
        if hasattr(model, 'module'):
            model = self.model.module
        self.prepare(model)
    
    def after_test_epoch(self, runner: "Runner"):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        self.analyze()
        self.dump()

    def after_test_iter(self, runner: "Runner", batch_idx: int, data_batch: dict, outputs: Any):
        if not torch.cuda.is_available():
            return
        if not self.on:
            return
        if self.sample_num is not None and batch_idx >= self.sample_num - 1:
            if hasattr(runner.test_loop, 'stop_testing'):
                runner.test_loop.stop_testing = True
