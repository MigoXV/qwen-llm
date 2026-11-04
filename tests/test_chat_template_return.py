"""Chat-template return compatibility and literal vision token regression tests."""
import asyncio
from collections import UserDict
from types import SimpleNamespace
import unittest

from qwen_llm.config import InferenceConfig
from qwen_llm.engine.generation import AsyncGenerationEngine

PAD = '<|image_pad|>'
PLACEHOLDER = '<|vision_start|>' + PAD + '<|vision_end|>'


class ChatTemplateReturnTests(unittest.TestCase):
    def engine(self, *, drop=False):
        self.encoded_text = None
        self.calls = []
        def template(messages, **kwargs):
            self.calls.append(kwargs)
            text = '\n'.join(m['content'] for m in messages)
            text += str(kwargs.get('tools', ''))
            if not kwargs.get('tokenize', True):
                return '' if drop else text
            ids = [1, 2]
            return UserDict(input_ids=ids) if kwargs.get('return_dict', True) else ids

        def encode(text, **kwargs):
            self.assertEqual(kwargs, {'add_special_tokens': False})
            self.encoded_text = text
            return [1] + [99] * text.count(PAD) + [2]

        engine = AsyncGenerationEngine(InferenceConfig())
        engine.tokenizer = SimpleNamespace(apply_chat_template=template, encode=encode)
        engine.image_placeholder = PLACEHOLDER
        engine.image_pad_token_id = 99
        engine.supports_tools = True
        return engine

    def test_text_requests_explicitly_return_token_list(self):
        result = asyncio.run(self.engine().prepare([{'role': 'user', 'content': 'hello'}], 'none'))
        self.assertEqual(result, [1, 2])
        self.assertIs(self.calls[0]['return_dict'], False)

    def test_two_images_with_twelve_literal_markers(self):
        engine = self.engine()
        images = [object(), object()]
        messages = [
            {'role': 'system', 'content': PLACEHOLDER * 4},
            {'role': 'assistant', 'content': PLACEHOLDER * 4},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'before'},
                {'type': 'image', 'image': images[0]},
                {'type': 'text', 'text': 'between'},
                {'type': 'image', 'image': images[1]}]},
        ]
        tools = [{'type': 'function', 'function': {'name': 'read', 'description': PLACEHOLDER * 4}}]
        result = asyncio.run(engine.prepare(messages, 'none', tools=tools))
        self.assertEqual(result['prompt_token_ids'], [1, 99, 99, 2])
        self.assertEqual(result['multi_modal_data']['image'], images)
        self.assertIn('before' + PLACEHOLDER + 'between' + PLACEHOLDER, self.encoded_text)
        self.assertEqual(self.encoded_text.count('&lt;|image_pad|&gt;'), 12)
        self.assertEqual(messages[0]['content'], PLACEHOLDER * 4)
        self.assertEqual(tools[0]['function']['description'], PLACEHOLDER * 4)

    def test_normal_image_requests(self):
        for count in (1, 2):
            with self.subTest(count=count):
                engine = self.engine()
                images = [object() for _ in range(count)]
                result = asyncio.run(engine.prepare([{'role': 'user', 'content': [
                    {'type': 'image', 'image': image} for image in images]}], 'none'))
                self.assertEqual(result['prompt_token_ids'], [1] + [99] * count + [2])
                self.assertEqual(result['multi_modal_data']['image'], images)

    def test_dropped_actual_image_is_rejected(self):
        engine = self.engine(drop=True)
        with self.assertRaisesRegex(ValueError, 'chat 模板'):
            asyncio.run(engine.prepare([{'role': 'user', 'content': [
                {'type': 'image', 'image': object()}]}], 'none'))


if __name__ == '__main__':
    unittest.main()
