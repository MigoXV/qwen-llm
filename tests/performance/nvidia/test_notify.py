"""真实 CPU IPC 回归：忙循环跨清理边界、空闲唤醒和阻塞取消。"""
import gc
from importlib.metadata import PackageNotFoundError, version
import multiprocessing as mp
import threading
import time

import pytest

try:
    supported = version('vllm') == '0.32.0'
except PackageNotFoundError:
    supported = False
pytestmark = pytest.mark.skipif(not supported, reason='仅验证独立 NVIDIA vLLM 0.32.0 环境')


def consume(handle, results, cancel):
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    reader = MessageQueue.create_from_handle(handle, 0)
    reader.wait_until_ready()
    reader._spin_condition.busy_loop_s = 0.05
    try:
        if cancel:
            timer = threading.Timer(0.2, reader.shutdown)
            timer.start()
            try:
                reader.dequeue(indefinite=True)
            except RuntimeError as error:
                results.send(str(error))
            finally:
                timer.join()
        else:
            values = [reader.dequeue(timeout=10) for _ in range(1025)]
            results.send(values)
    finally:
        results.close()
        del reader
        gc.collect()


@pytest.mark.parametrize('cancel', [False, True])
def test_shm_busy_idle_and_cancel(cancel):
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    context = mp.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    writer = MessageQueue(1, 1, max_chunk_bytes=4096, max_chunks=2)
    child = context.Process(target=consume, args=(writer.export_handle(), send, cancel))
    child.start()
    try:
        writer.wait_until_ready()
        if not cancel:
            for i in range(1025):
                if i == 512:
                    time.sleep(0.15)
                writer.enqueue({'sequence': i, 'text': f'通知-{i}'}, timeout=10)
        assert receive.poll(20), '读者未正常完成或取消'
        result = receive.recv()
        if cancel:
            assert result == 'cancelled'
        else:
            assert result == [{'sequence': i, 'text': f'通知-{i}'} for i in range(1025)]
        child.join(timeout=20)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        receive.close()
        send.close()
        del writer
        gc.collect()


def consume_memory(handle, results):
    """不保留消息对象，直接比较忙循环前后的 glibc 原生在用内存。"""
    import ctypes
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

    class MallInfo(ctypes.Structure):
        _fields_ = [(name, ctypes.c_size_t) for name in (
            'arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks',
            'fsmblks', 'uordblks', 'fordblks', 'keepcost')]

    libc = ctypes.CDLL(None)
    libc.mallinfo2.restype = MallInfo
    reader = MessageQueue.create_from_handle(handle, 0)
    reader.wait_until_ready()
    try:
        for i in range(61024):
            assert reader.dequeue(timeout=20) == i
            if i == 1023:
                before = libc.mallinfo2().uordblks
        results.send(libc.mallinfo2().uordblks - before)
    finally:
        results.close()
        del reader
        gc.collect()


def test_shm_notification_memory_bounded():
    import ctypes
    if not hasattr(ctypes.CDLL(None), 'mallinfo2'):
        pytest.skip('此原生内存回归要求 glibc mallinfo2')
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
    context = mp.get_context('spawn')
    receive, send = context.Pipe(duplex=False)
    writer = MessageQueue(1, 1, max_chunk_bytes=4096, max_chunks=2)
    child = context.Process(target=consume_memory, args=(writer.export_handle(), send))
    child.start()
    try:
        writer.wait_until_ready()
        for i in range(61024):
            writer.enqueue(i, timeout=20)
            # 让 PUB I/O 线程处理通知，避免极快发送时丢通知而掩盖积压。
            time.sleep(0.00015)
        assert receive.poll(20), '内存对照未完成'
        growth = receive.recv()
        child.join(timeout=20)
        assert child.exitcode == 0
        # 只约束增量，允许分配器预热；原始版本在同一实验增长约 533 KiB。
        assert growth < 192 * 1024, f'通知积压，原生堆增长 {growth} bytes'
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
        receive.close()
        send.close()
        del writer
        gc.collect()
