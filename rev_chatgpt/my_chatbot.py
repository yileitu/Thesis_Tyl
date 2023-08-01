# -*- coding: utf-8 -*-
from __future__ import annotations

from revChatGPT.V1 import Chatbot
from datetime import datetime

current_date = datetime.now()
current_date = current_date.strftime("%Y-%m-%d")
SYS_PROMPT = f"You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture. \n\
Knowledge cutoff: 2021-09 \n\
Current date: {current_date}"


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
		for _ in self.ask(prompt=SYS_PROMPT, auto_continue=True):
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
