from players.guesser import Guesser
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from model_manager import get_glove_model

    
class GuesserSBERT(Guesser):
    def __init__(
        self,
        team="Red",
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.3,
        avoid_penalty=1.0
    ):
        super().__init__()
        self.team = team
        self.clue = None
        self.num = 0
        self.guesses = 0
        self.similarity_threshold = similarity_threshold
        self.avoid_penalty = avoid_penalty
        self.guesses_this_turn = []
        
        # Load SBERT model
        print(f"Loading SBERT model '{model_name}'...")
        self.model = SentenceTransformer(model_name)

    def set_board(self, words):
        self.words = words
        print("BOARD")
        print("_" * 60)
        for i in range(0, len(words), 5):
            print("  ".join(words[i:i+5]))
        print("_" * 60)

    def set_clue(self, clue, num):
        self.clue = clue.lower()
        self.num = num
        self.guesses = 0
        self.guesses_this_turn = []
        print(f"The clue is: {clue} {num}")
        return [clue, num]

    def keep_guessing(self):
        if self.guesses >= self.num:
            return False
        if self.guesses > 0 and hasattr(self, 'last_similarity') and self.last_similarity < 0.4:
            return False
        return True

    def get_remaining_options(self):
        return [word for word in self.words if not word.startswith("*")]

    def _cosine_similarity(self, v1, v2):
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def get_answer(self):
        remaining_words = self.get_remaining_options()
        if not remaining_words:
            print("No remaining words to guess.")
            return None

        clue_vector = self.model.encode(self.clue)

        scored_words = []
        for word in remaining_words:
            word_vector = self.model.encode(word)
            similarity = self._cosine_similarity(word_vector, clue_vector)

            # Adjust score based on similarity to previous guesses this turn
            if self.guesses_this_turn:
                prev_vectors = self.model.encode(self.guesses_this_turn)
                prev_sim = np.mean([self._cosine_similarity(word_vector, vec) for vec in prev_vectors])
                similarity += prev_sim * 0.2

            scored_words.append((word, similarity))

        if not scored_words:
            print("No valid words found in vocabulary. Picking random word.")
            guess = random.choice(remaining_words)
        else:
            scored_words.sort(key=lambda x: x[1], reverse=True)
            print(f"\nSimilarity ranking (clue: '{self.clue}'):")
            for i, (word, sim) in enumerate(scored_words[:5]):
                print(f" {word:<15} -> {sim:.4f}")
            best_word, best_similarity = scored_words[0]
            self.last_similarity = best_similarity
            guess = best_word

        print(f"\nGuessing: {guess}")
        self.guesses += 1
        self.guesses_this_turn.append(guess)
        return guess
