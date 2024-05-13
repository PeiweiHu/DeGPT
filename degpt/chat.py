"""
Interfaces to interact with various LLMs
"""

import json
import os
import atexit
from typing import Dict, Optional, List
from openai import OpenAI

model = 'gpt-3.5-turbo'


api_key = None
api_base = None
assert (api_key and api_base and "Setup your api_key and api_base first")
client = OpenAI(api_key=api_key, base_url=api_base)


class QueryChatGPT():
    """ Interface for interacting with ChatGPT

    """

    def __init__(self) -> None:
        self.chat_context: List[Dict[str, str]] = []
        self.chat_history: List[Dict[str, str]] = []
        self.temperature:float = 0.2
        self.use_history = False
        self.system_prompt: Optional[str] = None
        atexit.register(self.log_history)

    def clear(self):
        self.chat_context = []

    def set_history(self, open: bool) -> None:
        self.use_history = open

    def insert_system_prompt(self, system_prompt: str) -> None:
        """ add system_prompt in self.chat_context """

        if self.chat_context and self.chat_context[0]["role"] == "system":
            self.chat_context[0]['content'] = system_prompt
        else:
            self.chat_context.insert(0, {
                "role": "system",
                "content": system_prompt
            })

    def log_history(self, log_file: str = 'chat_log.json'):

        if not os.path.exists(log_file):
            with open(log_file, 'w') as w:
                json.dump([], w, indent=4)

        with open(log_file, 'r') as r:
            log = json.load(r)
        assert (isinstance(log, list))
        log.append(self.chat_history)
        with open(log_file, 'w') as w:
            json.dump(log, w, indent=4)

    def __query(self, prompt: str, model: str) -> Optional[str]:
        self.chat_context.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            messages=self.chat_context,  # type: ignore
            model=model,
            temperature=self.temperature,
        )
        response_content = str(response.choices[0].message.content)

        self.chat_context.append({
            "role": "assistant",
            "content": response_content
        })
        self.chat_history.append({
            "role": "assistant",
            "content": response_content
        })

        return response_content

    def query(self,
              prompt: str,
              *,
              model: str = model) -> Optional[str]:

        response = self.__query(prompt, model)
        if not self.use_history:
            self.clear()
        return response
