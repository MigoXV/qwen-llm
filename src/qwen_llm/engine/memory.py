"""兼容旧 Ascend 插件的显式 KV 容量预算。"""


def bounded_kv_memory(available: int, requested: int | None) -> int:
    if requested is None:
        return available
    if requested <= 0 or requested > available:
        raise ValueError(f'KV 预算 {requested} bytes 超过可用容量 {available} 或非正数')
    return requested
