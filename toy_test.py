# -*- coding: utf-8 -*-
import re
RESPONSE_SPLIT = "### Response:"

output = """
Below is an input that describes a task or asks a problem. Write a response that appropriately completes the request.

### Input:
What are the names of some famous actors that started their careers on Broadway?

### Response:
Some famous actors that started their careers on Broadway include Matthew Broderick, Nathan Lane, Meryl Streep, Christopher Walken, and Angela Lansbury.
"""
pattern = re.compile(
			r"<unk>|<pad>|<s>|</s>|\[PAD\]|<\|endoftext\|>|\[UNK\]|\[CLS\]|\[MASK\]|<\|startofpiece\|>|<\|endofpiece\|>|\[gMASK\]|\[sMASK\]"
			)

print(text)
