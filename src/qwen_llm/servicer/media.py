"""解析 Chat Completions 消息中的图像 content part：仅接受内联 data URI，不抓取远端资源。"""
from __future__ import annotations

import base64
import binascii
import io
import re

MAX_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGE_PIXELS = 32_000_000
ALLOWED_MIME = ('image/png', 'image/jpeg', 'image/webp')

_DATA_URI = re.compile(r'^data:([^;,\s]+)\s*;\s*base64,([\s\S]+)$')


def _content_parts(message):
    content = message.get('content')
    return content if isinstance(content, list) else []


def count_image_parts(messages) -> int:
    """统计已解析消息（normalized_messages 之后）里的图像 part 数。"""
    return sum(1 for message in messages
               for part in _content_parts(message) if part.get('type') == 'image')


def trim_image_history(messages: list[dict], max_images: int) -> list[dict]:
    """Keep this turn intact and fill remaining slots with the newest history images.

    A turn starts at the last user message, including any following tool results.
    If that turn exceeds the limit, leave it intact for the caller to reject.
    Input messages and image objects are never mutated.
    """
    if max_images < 1:
        raise ValueError('max_images must be positive')
    last_user = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get('role') == 'user'), None)
    if last_user is None:
        return messages
    current_count = count_image_parts(messages[last_user:])
    if current_count > max_images:
        return messages
    history = [(i, j) for i, message in enumerate(messages[:last_user])
               for j, part in enumerate(_content_parts(message)) if part.get('type') == 'image']
    drop_count = max(0, len(history) - (max_images - current_count))
    if not drop_count:
        return messages
    dropped = {position: number for number, position in enumerate(history[:drop_count], 1)}
    result = []
    for i, message in enumerate(messages):
        if i < last_user and isinstance(message.get('content'), list):
            parts = [({'type': 'text', 'text':
                       f'[历史图片 {dropped[i, j]} 本轮未发送；如需查看，请重新上传。]'}
                      if (i, j) in dropped else part)
                     for j, part in enumerate(message['content'])]
            result.append({**message, 'content': parts})
        else:
            result.append(message)
    return result


def parse_image_part(part: dict):
    """校验并解码单个 image content part，返回 RGB PIL.Image；失败抛中文说明的 ValueError。"""
    if set(part) - {'type', 'image_url'} or part.get('type') != 'image_url':
        raise ValueError('image content part 只支持 type 与 image_url 字段')
    image_url = part.get('image_url')
    if not isinstance(image_url, dict) or not isinstance(image_url.get('url'), str):
        raise ValueError('image_url.url 必须是字符串')
    match = _DATA_URI.match(image_url['url'].strip())
    if match is None:
        raise ValueError('图像只接受内联 data URI，不支持远端 URL')
    mime = match.group(1).lower()
    if mime not in ALLOWED_MIME:
        raise ValueError('图像格式不受支持（仅支持 png/jpeg/webp）')
    payload = ''.join(match.group(2).split())
    if len(payload) > 4 * (MAX_IMAGE_BYTES // 3 + 1):
        raise ValueError(f'图像不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB')
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError('图像 base64 编码不合法') from exc
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f'图像不能超过 {MAX_IMAGE_BYTES // 1024 // 1024} MB')
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError('图像内容无法解码') from exc
    if image.format not in {'PNG', 'JPEG', 'WEBP'} or image.width * image.height > MAX_IMAGE_PIXELS:
        raise ValueError('图像像素过大或格式不受支持')
    return image.convert('RGB')
