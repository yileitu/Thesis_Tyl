# -*- coding: utf-8 -*-
from __future__ import annotations

from revChatGPT.V1 import Chatbot
from datetime import datetime

from util.constants import CHATGPT_SYS_PROMPT


class MyChatbot(Chatbot):
	def __init__(
			self,
			config: dict[str, str],
			conversation_id: str | None = None,
			parent_id: str | None = None,
			lazy_loading: bool = True,
			base_url: str | None = None,
			) -> None:
		super().__init__(config, conversation_id, parent_id, lazy_loading, base_url)
		for _ in self.ask(prompt=CHATGPT_SYS_PROMPT, auto_continue=True):
			pass

	def get_response(self, prompt: str) -> str:
		"""
		Get response from revChatGPT

		:param prompt: prompt
		:return: response
		"""
		response = ""
		for data in self.ask(prompt=prompt, auto_continue=True):
			response = data["message"]

		return response
