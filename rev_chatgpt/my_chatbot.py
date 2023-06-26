# -*- coding: utf-8 -*-
from __future__ import annotations

from revChatGPT.V1 import Chatbot


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

	def get_response(self, prompt: str) -> str:
		"""
		Get response from revChatGPT

		:param prompt: prompt
		:return: response
		"""
		response = ""
		for data in self.ask(prompt):
			response = data["message"]

		return response
