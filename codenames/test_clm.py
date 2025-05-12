from local_model_clm import LocalLLM_CLM
from huggingface_hub import login


#hugging face token cannot be pushed to git

llm = LocalLLM_CLM()

prompt = (
    "You are a Codenames Codemaster. Give a one-word clue that connects your team's words and avoids opponent/neutral words.\n"
    "Include a number showing how many team words the clue relates to.\n\n"
    "Also I want you to explain the last clue \n"
    "Examples:"
    "Team words: apple, banana, water\n"
    "Avoid: grenade, bomb, pit\n"
    "Clue: fruit 2\n\n"
    "Team words: eagle, falcon, hawk\n"
    "Avoid: jet, pilot, wing\n"
    "Clue: raptor 3\n\n"
    "End of the examples now you try to give a nice clue \n\n"
    "Team words: sun, moon, star, sky, wind,fire\n"
    "Avoid: cloud, rain, snow,saturn\n"
    "Clue:"
)

output = llm.generate(prompt)
print("Model Output:", output)