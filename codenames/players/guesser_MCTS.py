from players.guesser import Guesser
import random
import math
import numpy as np
from model_manager import get_glove_model

class GuesserMCTS(Guesser):
    """
    Guesser that uses Monte Carlo Tree Search to evaluate guess sequences
    """
    
    def __init__(self, team="Red", model_name="glove-wiki-gigaword-300", 
                 num_simulations=100, exploration_weight=1.4):
        super().__init__()
        self.team = team
        # Use shared model instead of loading new instance
        print(f"Using shared {model_name} model...")
        self.model = get_glove_model(model_name)
        
        # MCTS parameters
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        
        # Game state
        self.words = []
        self.clue = None
        self.num = 0
        self.guesses_made = 0
        self.guesses_this_turn = []
        
        # Probabilities for different card types (will be updated based on game state)
        self.team_prob = 0.36  # 9/25 for red, 8/25 for blue
        self.opponent_prob = 0.32  # 8/25 for red, 9/25 for blue  
        self.civilian_prob = 0.28  # 7/25
        self.assassin_prob = 0.04  # 1/25
    
    def set_board(self, words):
        """Set the current board state"""
        self.words = words
        # Update probabilities based on remaining cards
        remaining_words = [w for w in words if not w.startswith("*")]
        if remaining_words:
            total_remaining = len(remaining_words)
            self.team_prob = max(0.2, min(0.5, len(remaining_words) * 0.36 / 25))
            self.opponent_prob = max(0.2, min(0.5, len(remaining_words) * 0.32 / 25))
            self.civilian_prob = max(0.1, min(0.4, len(remaining_words) * 0.28 / 25))
            self.assassin_prob = max(0.02, min(0.1, len(remaining_words) * 0.04 / 25))
    
    def set_clue(self, clue, num):
        """Set the current clue and reset turn state"""
        self.clue = clue.lower()
        self.num = num
        self.guesses_made = 0
        self.guesses_this_turn = []
        print(f"MCTS Guesser received clue: {clue} {num}")
        return [clue, num]
    
    def keep_guessing(self):
        """Decide whether to keep guessing using MCTS simulation"""
        if self.guesses_made >= self.num + 1:  # Can't guess more than num + 1
            return False
        
        if self.guesses_made == 0:  # Must make at least one guess
            return True
        
        remaining_words = self.get_remaining_options()
        if not remaining_words:
            return False
        
        # Quick MCTS simulation to decide if continuing is worth it
        continue_value = self._evaluate_continue_guessing(remaining_words)
        stop_value = 0  # Baseline value of stopping
        
        return continue_value > stop_value
    
    def get_remaining_options(self):
        """Get words that haven't been guessed yet"""
        return [word for word in self.words if not word.startswith("*")]
    
    def _word_similarity(self, w1, w2):
        """Calculate cosine similarity between two words"""
        w1, w2 = w1.lower(), w2.lower()
        if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
            return 0.0
        v1 = self.model[w1]
        v2 = self.model[w2]
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def _simulate_guess_outcome(self, word):
        """Simulate the outcome of guessing a word"""
        
        rand = random.random()
        if rand < self.team_prob:
            return "team"  # Correct team word
        elif rand < self.team_prob + self.opponent_prob:
            return "opponent"  # Opponent's word
        elif rand < self.team_prob + self.opponent_prob + self.civilian_prob:
            return "civilian"  # Civilian word
        else:
            return "assassin"  # Assassin word
    
    def _evaluate_guess_sequence(self, words, max_depth=3):
        """Evaluate a sequence of guesses using simulation"""
        if not words or max_depth <= 0:
            return 0.0
        
        total_reward = 0.0
        num_simulations = 20  # Reduced for efficiency
        
        for _ in range(num_simulations):
            reward = 0.0
            guesses_made = 0
            
            for word in words[:max_depth]:
                outcome = self._simulate_guess_outcome(word)
                guesses_made += 1
                
                if outcome == "team":
                    reward += 2.0  # Good guess
                    # Continue guessing (in real game, this would be allowed)
                elif outcome == "opponent":
                    reward -= 1.0  # Bad guess, turn ends
                    break
                elif outcome == "civilian":
                    reward -= 0.5  # Neutral guess, turn ends
                    break
                elif outcome == "assassin":
                    reward -= 10.0  # Game over
                    break
            
            total_reward += reward
        
        return total_reward / num_simulations
    
    def _evaluate_continue_guessing(self, remaining_words):
        """Evaluate the value of continuing to guess"""
        if not remaining_words:
            return -1.0
        
        # Rank words by similarity to clue
        word_similarities = []
        for word in remaining_words:
            if word.lower() not in [g.lower() for g in self.guesses_this_turn]:
                sim = self._word_similarity(self.clue, word)
                word_similarities.append((word, sim))
        
        word_similarities.sort(key=lambda x: x[1], reverse=True)
        top_words = [w for w, _ in word_similarities[:3]]  # Consider top 3
        
        # Simulate guessing these words
        return self._evaluate_guess_sequence(top_words, max_depth=2)
    
    class MCTSNode:
        """Node in the MCTS tree for guess evaluation"""
        def __init__(self, word=None, parent=None):
            self.word = word
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.untried_words = []
        
        def uct_value(self, exploration_weight, parent_visits):
            """Calculate UCT value for node selection with proper bounds checking"""
            if self.visits == 0:
                return float('inf')
            
            # Ensure parent_visits is positive to avoid log domain error
            if parent_visits <= 0:
                return float('inf')
            
            exploitation = self.value / self.visits
            
            # Proper UCT calculation with bounds checking
            try:
                log_term = math.log(parent_visits)
                if log_term <= 0:
                    exploration = 0
                else:
                    exploration = exploration_weight * math.sqrt(log_term / self.visits)
            except (ValueError, ZeroDivisionError):
                exploration = 0
            
            return exploitation + exploration
                
        def best_child(self, exploration_weight):
            """Select best child based on UCT value with error handling"""
            if not self.children:
                return None
            
            try:
                return max(self.children, 
                        key=lambda c: c.uct_value(exploration_weight, self.visits))
            except (ValueError, TypeError):
                # Fallback to random selection if UCT calculation fails
                return random.choice(self.children) if self.children else None
        
        def expand(self, word):
            child = GuesserMCTS.MCTSNode(word, self)
            self.children.append(child)
            return child
        
        def is_fully_expanded(self):
            return not self.untried_words
    
    def _run_mcts_for_guess(self, available_words):
        """Run MCTS to find the best word to guess"""
        if not available_words:
            return None
        
        # Create root node
        root = self.MCTSNode()
        root.untried_words = available_words.copy()
        
        # Run MCTS simulations
        for _ in range(self.num_simulations):
            # Selection
            node = root
            path = []
            
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_weight)
                path.append(node)
            
            # Expansion
            if not node.is_fully_expanded():
                word = random.choice(node.untried_words)
                node.untried_words.remove(word)
                node = node.expand(word)
                path.append(node)
            
            # Simulation
            # Create a guess sequence starting with this node's word
            guess_sequence = [n.word for n in path if n.word]
            reward = self._evaluate_guess_sequence(guess_sequence)
            
            # Backpropagation
            for node in reversed(path):
                node.visits += 1
                node.value += reward
        
        # Select best child
        if root.children:
            best_child = max(root.children, key=lambda c: c.value / max(1, c.visits))
            return best_child.word
        
        return None
    
    def get_answer(self):
        """Get the next guess using MCTS"""
        remaining_words = self.get_remaining_options()
        
        if not remaining_words:
            print("No remaining words to guess.")
            return None
        
        # Filter out words already guessed this turn
        available_words = [w for w in remaining_words 
                          if w.lower() not in [g.lower() for g in self.guesses_this_turn]]
        
        if not available_words:
            print("No new words available to guess.")
            return None
        
        # If no clue or clue not in vocabulary, use similarity-based approach
        if not self.clue or self.clue not in self.model.key_to_index:
            print("Clue not in vocabulary, using similarity-based fallback.")
            # Fallback to similarity-based selection
            word_similarities = []
            for word in available_words:
                if word.lower() in self.model.key_to_index:
                    # Use average similarity to previous guesses as a heuristic
                    if self.guesses_this_turn:
                        avg_sim = np.mean([self._word_similarity(word, prev) 
                                         for prev in self.guesses_this_turn])
                        word_similarities.append((word, avg_sim))
                    else:
                        # Random selection for first guess without valid clue
                        word_similarities.append((word, random.random()))
            
            if word_similarities:
                word_similarities.sort(key=lambda x: x[1], reverse=True)
                guess = word_similarities[0][0]
            else:
                guess = random.choice(available_words)
        else:
            # Use MCTS to select the best word
            guess = self._run_mcts_for_guess(available_words)
            
            # Fallback if MCTS fails
            if not guess:
                # Use similarity to clue as fallback
                word_similarities = [(w, self._word_similarity(self.clue, w)) 
                                   for w in available_words]
                word_similarities.sort(key=lambda x: x[1], reverse=True)
                guess = word_similarities[0][0] if word_similarities else available_words[0]
        
        # Update state
        self.guesses_made += 1
        self.guesses_this_turn.append(guess)
        
        print(f"MCTS Guesser selected: {guess}")
        return guess