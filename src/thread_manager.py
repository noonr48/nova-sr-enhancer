#!/usr/bin/env python3
"""
Thread Manager - CPU affinity and thread pool management

Provides optimized thread management for NovaSR audio processing.
Includes CPU affinity pinning and lock-free queues.
"""

import os
import threading
import queue
import ctypes
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Callable, Any, Dict, List
from functools import partial
import time


def set_cpu_affinity(cpu_id: int):
    """
    Set thread CPU affinity to a specific core.

    Args:
        cpu_id: CPU core ID (0-based)
    """
    try:
        # Get current thread ID
        tid = threading.get_ident()

        # Try using pthread_setaffinity_np (Linux)
        libc = ctypes.CDLL("libc.so.6")
        pthread_setaffinity_np = libc.pthread_setaffinity_np

        # CPU set size (1024 bits = 128 bytes)
        cpu_set_t = ctypes.c_ubyte * 128
        cpuset = cpu_set_t()

        # Set the bit for the specified CPU
        cpu_id_byte = cpu_id // 8
        cpu_id_bit = cpu_id % 8
        cpuset[cpu_id_byte] |= (1 << cpu_id_bit)

        pthread_setaffinity_np(tid, 128, ctypes.byref(cpuset))

    except Exception as e:
        # Fallback: ignore if affinity setting fails
        pass


def set_cpu_affinity_range(cpu_ids: List[int]):
    """
    Set thread CPU affinity to a range of cores.

    Args:
        cpu_ids: List of CPU core IDs
    """
    try:
        tid = threading.get_ident()
        libc = ctypes.CDLL("libc.so.6")
        pthread_setaffinity_np = libc.pthread_setaffinity_np

        cpu_set_t = ctypes.c_ubyte * 128
        cpuset = cpu_set_t()

        for cpu_id in cpu_ids:
            cpu_id_byte = cpu_id // 8
            cpu_id_bit = cpu_id % 8
            cpuset[cpu_id_byte] |= (1 << cpu_id_bit)

        pthread_setaffinity_np(tid, 128, ctypes.byref(cpuset))

    except Exception:
        pass


def get_cpu_count() -> int:
    """Get number of CPU cores."""
    return multiprocessing.cpu_count()


class BoundedQueue:
    """
    Lock-free-ish bounded queue using standard queue with optimized sizing.

    Uses Python's queue.Queue but with optimized configuration
    for audio processing pipelines.
    """

    def __init__(self, maxsize: int = 16):
        self._queue = queue.Queue(maxsize=maxsize)
        self.maxsize = maxsize

    def put(self, item, block: bool = True, timeout: Optional[float] = None):
        """Put item in queue."""
        return self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None):
        """Get item from queue."""
        return self._queue.get(block=block, timeout=timeout)

    def put_nowait(self, item):
        """Put item without blocking."""
        return self._queue.put_nowait(item)

    def get_nowait(self):
        """Get item without blocking."""
        return self._queue.get_nowait()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    def qsize(self) -> int:
        """Get approximate queue size."""
        return self._queue.qsize()

    def clear(self):
        """Clear all items from queue."""
        while not self.empty():
            try:
                self.get_nowait()
            except queue.Empty:
                break


class ThreadManager:
    """
    Manages thread pools with CPU affinity for audio processing.

    Typical usage for 6 cores / 12 threads:
    - Thread 0: Audio capture (Core 0)
    - Thread 1-2: NovaSR processing (Cores 1-2)
    - Thread 3-4: NovaSR processing (Cores 3-4)
    - Thread 5: Audio playback (Core 5)
    - Threads 6+: Batch processing pool (all cores)
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        enable_affinity: bool = True
    ):
        """
        Initialize thread manager.

        Args:
            max_workers: Maximum number of worker threads (None = CPU count)
            enable_affinity: Enable CPU affinity pinning
        """
        self.cpu_count = get_cpu_count()
        self.max_workers = max_workers or self.cpu_count
        self.enable_affinity = enable_affinity

        # Reserved cores for specific tasks (on 6-core system)
        # Core 0: Capture/IO
        # Cores 1-4: Processing
        # Core 5: Playback/IO
        self.captue_core = 0
        self.playback_core = self.cpu_count - 1
        self.processing_cores = list(range(1, self.cpu_count - 1))

        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: List[Future] = []

    def get_processing_threads(self) -> List[int]:
        """Get CPU cores for processing threads."""
        return self.processing_cores

    def create_executor(
        self,
        worker_name: str = "nova-sr-worker"
    ) -> ThreadPoolExecutor:
        """
        Create thread pool executor with CPU affinity.

        Args:
            worker_name: Name prefix for worker threads

        Returns:
            ThreadPoolExecutor instance
        """
        if self._executor is not None:
            return self._executor

        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix=worker_name
        )

        return self._executor

    def submit(
        self,
        fn: Callable,
        *args,
        cpu_affinity: Optional[int] = None,
        **kwargs
    ) -> Future:
        """
        Submit task to thread pool with optional CPU affinity.

        Args:
            fn: Function to execute
            *args: Function arguments
            cpu_affinity: Specific CPU core to pin thread to
            **kwargs: Function keyword arguments

        Returns:
            Future for the task
        """
        if self._executor is None:
            self.create_executor()

        def wrapped_fn():
            if self.enable_affinity and cpu_affinity is not None:
                set_cpu_affinity(cpu_affinity)
            return fn(*args, **kwargs)

        future = self._executor.submit(wrapped_fn)
        self._futures.append(future)
        return future

    def submit_batch(
        self,
        fn: Callable,
        items: List[Any],
        affinity_list: Optional[List[int]] = None
    ) -> List[Future]:
        """
        Submit multiple tasks for batch processing.

        Args:
            fn: Function to execute for each item
            items: List of items to process
            affinity_list: CPU affinity for each task (None = round-robin)

        Returns:
            List of Future objects
        """
        futures = []

        for i, item in enumerate(items):
            if affinity_list and i < len(affinity_list):
                affinity = affinity_list[i]
            else:
                # Round-robin through processing cores
                affinity = self.processing_cores[i % len(self.processing_cores)]

            future = self.submit(fn, item, cpu_affinity=affinity)
            futures.append(future)

        return futures

    def wait_all(self, futures: Optional[List[Future]] = None, timeout: Optional[float] = None):
        """
        Wait for all futures to complete.

        Args:
            futures: List of futures to wait for (None = all submitted)
            timeout: Maximum time to wait per future
        """
        if futures is None:
            futures = self._futures

        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(f"[ThreadManager] Task error: {e}")

    def shutdown(self, wait: bool = True):
        """
        Shutdown thread pool.

        Args:
            wait: Wait for pending tasks to complete
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
        self._futures.clear()


class AffinityThread(threading.Thread):
    """
    Thread with automatic CPU affinity pinning.
    """

    def __init__(
        self,
        target: Callable,
        cpu_affinity: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize affinity thread.

        Args:
            target: Function to run in thread
            cpu_affinity: CPU core to pin to
            name: Thread name
            **kwargs: Additional Thread arguments
        """
        self._cpu_affinity = cpu_affinity
        self._target_fn = target
        super().__init__(target=self._wrapped_target, name=name, **kwargs)

    def _wrapped_target(self):
        """Wrapper that sets CPU affinity before running target."""
        if self._cpu_affinity is not None:
            set_cpu_affinity(self._cpu_affinity)
        return self._target_fn()


# CPU info utility
def print_cpu_info():
    """Print CPU information."""
    cpu_count = get_cpu_count()
    print(f"[CPU] Detected {cpu_count} cores")

    # Try to get CPU model
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    print(f"[CPU] {line.split(':', 1)[1].strip()}")
                    break
    except Exception:
        pass

    print(f"[CPU] Affinity available: {hasattr(ctypes.CDLL('libc.so.6'), 'pthread_setaffinity_np')}")


if __name__ == "__main__":
    print_cpu_info()

    # Test thread manager
    print("\n[Test] Creating thread manager...")
    manager = ThreadManager(max_workers=4)
    print(f"[Test] Processing cores: {manager.get_processing_threads()}")

    # Test affinity thread
    print("\n[Test] Creating affinity thread...")

    def worker():
        import time
        time.sleep(0.1)
        return f"Thread {threading.get_ident()}"

    thread = AffinityThread(target=worker, cpu_affinity=1, name="test-worker")
    thread.start()
    thread.join()
    print(f"[Test] Thread result: {thread}")

    # Test batch submission
    print("\n[Test] Submitting batch tasks...")

    def process_item(x):
        import time
        time.sleep(0.05)
        return x * 2

    items = list(range(8))
    futures = manager.submit_batch(process_item, items)
    results = [f.result() for f in futures]
    print(f"[Test] Results: {results}")

    manager.shutdown()
    print("[Test] Done!")
