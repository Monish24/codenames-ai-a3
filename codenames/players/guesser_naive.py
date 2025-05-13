from players.guesser import Guesser
import random
import numpy as np
from gensim.downloader import load as gensim_load


class AIGuesser(Guesser):
    def __init__(self, team="Red", model_name="glove-wiki-gigaword-300"):
        super().__init__()
        self.team = team
        self.clue = None
        self.num = 0
        self.guesses = 0

        # Load embedding model
        print(f"Loading embedding model on guesser {team} '{model_name}'...")
        self.model = gensim_load(model_name)

    def set_board(self, words):
        self.words = words

    def set_clue(self, clue, num):
        self.clue = clue.lower()
        self.num = num
        self.guesses = 0
        print("The clue is:", clue, num)
        return [clue, num]

    def keep_guessing(self):
        # Guess up to num + 1
        print(f'Guesses: {self.guesses} codemaster num: {self.num}')
        return self.guesses < self.num

    def get_remaining_options(self):
        return [word for word in self.words if not word.startswith("*")]

    def get_answer(self):
        remaining_words = self.get_remaining_options()
        # print(f"Remaining options: {remaining_words}")

        valid_words = [word.lower() for word in remaining_words if word.lower() in self.model.key_to_index]
        if self.clue not in self.model.key_to_index:
            print("Clue not found in vocabulary, picking random word.")
            guess = random.choice(remaining_words)
        else:
            clue_vector = self.model[self.clue]
            sims = []
            for word in valid_words:
                sim = np.dot(self.model[word], clue_vector)
                sims.append((word, sim))

            if sims:
                # Sort similarities
                sims.sort(key=lambda x: x[1], reverse=True)
                print("\nSimilarity ranking (clue: '{}'):\n".format(self.clue))
                for word, sim in sims:
                    print(f"  {word:<15} -> {sim:.4f}")

                best_word = sims[0][0]
                # Match to original-case word in remaining_words
                guess = next(w for w in remaining_words if w.lower() == best_word)
            else:
                print("No valid words found in vocabulary. Picking random word.")
                guess = random.choice(remaining_words)

        self.guesses += 1
        print(f"\nGuessing: {guess}")
        return guess