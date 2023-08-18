from dataclasses import dataclass
from enum import IntEnum


@dataclass
class MCOptions:
	"""
	Option texts for multiple choice questions
	"""
	A: str = ''
	B: str = ''
	C: str = ''
	D: str = ''
	E: str = ''


class Task(IntEnum):
	"""
	Enum for task types
	"""
	QA: int = 0  # Question Answering
	MC: int = 1  # Multiple Choice
	TF: int = 2  # True/False
	EXAM: int = 3  # MC + TF
	TOY_MC: int = 4  # Toy task for debugging, MC only
