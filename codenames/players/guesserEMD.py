from players.guesser import Guesser
import random
import numpy as np
from gensim.models import KeyedVectors
from model_manager import get_glove_model

class GuesserEmbeddings(Guesser):
    """
    An improved guesser that uses word embeddings with multiple strategies
    for making educated guesses.
    """
    
    def __init__(
        self,
        team="Red",
        model_name="glove-wiki-gigaword-300",
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
        
        # Use shared model instead of loading new instance
        print(f"Using shared {model_name} model...")
        self.model = get_glove_model(model_name)
    
    def set_board(self, words):
        """
        Set the current board state and print it in a format the GUI can recognize.
        """
        self.words = words
        
        # Print board state in a format the GUI can recognize
        print("BOARD")
        print("_" * 60)
        for i in range(0, len(words), 5):
            print("  ".join(words[i:i+5]))
        print("_" * 60)
    
    def set_clue(self, clue, num):
        """
        Set the current clue and number of words to guess.
        """
        self.clue = clue.lower()
        self.num = num
        self.guesses = 0
        self.guesses_this_turn = []
        print(f"The clue is: {clue} {num}")
        return [clue, num]
    
    def keep_guessing(self):
        """
        Determine whether to keep guessing based on the current state.
        Uses a more strategic approach to decide whether to continue guessing.
        """
        # If we've already guessed as many as the clue number, be more cautious
        if self.guesses >= self.num:
            return False
        
        # If we've had a very confident guess and now confidence drops, stop
        if self.guesses > 0 and hasattr(self, 'last_similarity') and self.last_similarity < 0.4:
            return False
            
        return True
    
    def get_remaining_options(self):
        """
        Get the list of words that haven't been guessed yet.
        """
        return [word for word in self.words if not word.startswith("*")]
    
    def _word_similarity(self, w1, w2):
        """
        Calculate cosine similarity between two words.
        Returns 0.0 if either word is not in the vocabulary.
        """
        w1, w2 = w1.lower(), w2.lower()
        if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
            return 0.0
        v1 = self.model[w1]
        v2 = self.model[w2]
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def get_answer(self):
        """
        Choose the best word based on clue similarity and previous guesses.
        """
        remaining_words = self.get_remaining_options()
        
        if not remaining_words:
            print("No remaining words to guess.")
            return None
        
        # Check if clue is in vocabulary
        if self.clue not in self.model.key_to_index:
            print("Clue not found in vocabulary, picking random word.")
            guess = random.choice(remaining_words)
            self.guesses += 1
            self.guesses_this_turn.append(guess)
            return guess
        
        # Calculate similarities with adjustment for previous guesses
        scored_words = []
        clue_vector = self.model[self.clue]
        
        for word in remaining_words:
            word_lower = word.lower()
            
            # Skip words not in model vocabulary
            if word_lower not in self.model.key_to_index:
                continue
                
            word_vector = self.model[word_lower]
            
            # Calculate similarity
            similarity = np.dot(word_vector, clue_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(clue_vector))
            
            # Adjust score based on similarity to previous guesses this turn
            similarity_to_prev_guesses = 0
            if self.guesses_this_turn:
                for prev_guess in self.guesses_this_turn:
                    prev_lower = prev_guess.lower()
                    if prev_lower in self.model.key_to_index:
                        similarity_to_prev_guesses += self._word_similarity(word_lower, prev_lower)
                similarity_to_prev_guesses /= len(self.guesses_this_turn)
                
                # Bonus for words similar to previous successful guesses
                similarity += similarity_to_prev_guesses * 0.2
            
            scored_words.append((word, similarity))
        
        if not scored_words:
            print("No valid words found in vocabulary. Picking random word.")
            guess = random.choice(remaining_words)
        else:
            # Sort by similarity score
            scored_words.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nSimilarity ranking (clue: '{self.clue}'):")
            for i, (word, sim) in enumerate(scored_words[:5]):  # Show top 5
                print(f" {word:<15} -> {sim:.4f}")
            
            best_word, best_similarity = scored_words[0]
            self.last_similarity = best_similarity
            guess = best_word
        
        # Print the guess in a consistent format for the GUI to recognize
        print(f"\nGuessing: {guess}")
        
        self.guesses += 1
        self.guesses_this_turn.append(guess)
        return guess