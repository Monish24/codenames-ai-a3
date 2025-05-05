# test_flan.py
from local_model import LocalLLM

llm = LocalLLM()
print(llm.ask("Give a one-word clue for 'apple', 'banana', and 'grape'"))
