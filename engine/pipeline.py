import asyncio
from typing import Callable, Dict, Any, List, Set
from collections import defaultdict, deque

class AsyncToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str = None):
        def decorator(func: Callable):
            tool_name = name or func.__name__
            self.tools[tool_name] = func
            return func
        return decorator


class AsyncPipelineEngine:
    def __init__(self, registry: AsyncToolRegistry, max_concurrent_tasks: int = 5):
        """
        :param max_concurrent_tasks: Giới hạn số lượng task chạy song song cùng lúc (Pool Size)
        """
        self.registry = registry
        self.steps: Dict[str, Dict[str, Any]] = {}
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)
        
        # Pool giới hạn số lượng task chạy đồng thời để bảo vệ RAM
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

    def add_step(self, name: str, tool_name: str, depends_on: List[str] = None):
        depends_on = depends_on or []
        self.steps[name] = {
            "tool": tool_name,
            "depends_on": depends_on
        }
        
        for parent in depends_on:
            self.graph[parent].append(name)
            self.in_degree[name] += 1
            
        if name not in self.in_degree:
            self.in_degree[name] = 0

    async def _run_single_step_safe(self, step_name: str, state: Dict[str, Any], state_lock: asyncio.Lock):
        """Bọc execution trong Semaphore để đảm bảo không vượt quá Pool Capacity"""
        async with self.semaphore:  # 🛑 Chờ nếu Pool đã đầy
            step_info = self.steps[step_name]
            tool_fn = self.registry.tools.get(step_info["tool"])

            if not tool_fn:
                raise KeyError(f"Tool '{step_info['tool']}' chưa được đăng ký!")

            print(f"🚀 [START]: Step '{step_name}'")
            
            # Chạy tool
            if asyncio.iscoroutinefunction(tool_fn):
                result = await tool_fn(state)
            else:
                result = tool_fn(state)

            # Cập nhật state an toàn
            async with state_lock:
                if isinstance(result, dict):
                    state.update(result)
                elif result is not None:
                    state[step_name] = result

            print(f"✅ [DONE ]: Step '{step_name}'")
            return step_name

    async def run(self, initial_state: Dict[str, Any] = None) -> Dict[str, Any]:
        state = initial_state or {}
        in_degree = self.in_degree.copy()
        
        ready_queue = deque([node for node in self.steps if in_degree[node] == 0])
        completed_steps: Set[str] = set()
        state_lock = asyncio.Lock()

        while ready_queue:
            current_batch = list(ready_queue)
            ready_queue.clear()

            print(f"\n⚡ Batch ready: {len(current_batch)} tasks (Pool Limit: {self.semaphore._value} slot free)")
            
            # Chạy toàn bộ batch thông qua wrapper có Semaphore
            finished_batch = await asyncio.gather(*[
                self._run_single_step_safe(name, state, state_lock) 
                for name in current_batch
            ])

            for step_name in finished_batch:
                completed_steps.add(step_name)
                for neighbor in self.graph[step_name]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        ready_queue.append(neighbor)

        if len(completed_steps) != len(self.steps):
            raise ValueError("Lỗi: Phát hiện chu kỳ phụ thuộc (Circular Dependency)!")

        return state
