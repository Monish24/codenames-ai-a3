
from players.codemaster import Codemaster
import random
import math
import numpy as np
from model_manager import get_glove_model

class CodemasterMCTS(Codemaster):
    """
    True MCTS-based Codemaster that uses GloVe embeddings for semantic similarity.
    Simulates the game to find optimal clues for multiple words.
    """

    def __init__(
        self,
        team: str = None,
        model_name: str = "glove-wiki-gigaword-300",
        num_simulations: int = 200, 
        exploration_weight: float = 1.4,
    ):
        super().__init__()
        self.team = team
        # Use shared model instead of loading new instance
        print(f"Using shared {model_name} model...")
        self.model = get_glove_model(model_name)

        # MCTS parameters
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight

        # Game state placeholders
        self.words_on_board = []
        self.key_grid = []
        self.color = None
        self.my_words = []
        self.opponent_words = []
        self.civilian_words = []
        self.assassin_word = None

        self.win_reward = 10.0
        self.assassin_penalty = -100.0
        self.opponent_penalty = -5.0
        self.civilian_penalty = -1.0
        self.target_reward = 5.0

        vocab_items = sorted(self.model.key_to_index.items(),
                             key=lambda kv: kv[1])[:15000]  # More words for diversity
        self.vocabulary = [w for w,_ in vocab_items if w.isalpha() and len(w) >= 3]

        self.word_frequency = {}
        for i, (word, _) in enumerate(vocab_items):
            self.word_frequency[word] = 1.0 - (i / len(vocab_items))  # 1.0 = most frequent

    def set_game_state(self, words_on_board, key_grid):
        """
        Called each turn with the current board words and key‐grid.
        """
        self.words_on_board = [w.lower() for w in words_on_board]
        self.key_grid = key_grid

        self.color = self.team or next(
            (k for k in key_grid if k not in ("Civilian","Assassin")),
            "Red"
        )

        self.my_words = []
        self.opponent_words = []
        self.civilian_words = []
        self.assassin_word = None

        for w,k in zip(self.words_on_board, self.key_grid):
            if k == self.color:
                self.my_words.append(w)
            elif k == "Assassin":
                self.assassin_word = w
            elif k == "Civilian":
                self.civilian_words.append(w)
            else:
                self.opponent_words.append(w)

    def _word_similarity(self, w1, w2) -> float:
        """
        Cosine similarity between two words in the GloVe space.
        """
        w1, w2 = w1.lower(), w2.lower()
        if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
            return 0.0
        v1 = self.model[w1]
        v2 = self.model[w2]
        return float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))

    def _find_potential_clues(self, target_words, avoid_words, max_candidates=200):
        """
        Find potential clue words semantically related to target words.
        Returns a list of candidate clues sorted by their initial score.
        """
        candidates = {}
        
        for word in target_words:
            if word not in self.model.key_to_index:
                continue
                
            try:
                similar = self.model.most_similar(positive=[word], topn=50)
                for clue, sim in similar:
                    if clue not in candidates:
                        candidates[clue] = {'similarities': {}, 'score': 0.0}
                    candidates[clue]['similarities'][word] = sim
            except:
                continue
                
        # For multi-target clues
        if len(target_words) > 1:
            valid_targets = [w for w in target_words if w in self.model.key_to_index]
            if len(valid_targets) > 1:
                try:
                    # Calculate centroid of target words
                    vectors = [self.model[w] for w in valid_targets]
                    centroid = np.mean(vectors, axis=0)
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm > 0:
                        centroid = centroid / centroid_norm
                        
                        # Find words similar to centroid
                        centroid_similar = self.model.similar_by_vector(centroid, topn=100)
                        for clue, sim in centroid_similar:
                            if clue not in candidates:
                                candidates[clue] = {'similarities': {}, 'score': 0.0}
                            
                            # Add similarity to all target words
                            for word in valid_targets:
                                word_sim = self._word_similarity(clue, word)
                                candidates[clue]['similarities'][word] = word_sim
                except:
                    pass
        
        # Calculate preliminary scores - avoid invalid or risky clues
        board_set = set(w.lower() for w in self.words_on_board)
        filtered_candidates = {}
        
        for clue, data in candidates.items():
            # Skip non-alphabetic or too short words
            if not clue.isalpha() or len(clue) < 3:
                continue
                
            # Skip board words or derived forms
            if clue in board_set:
                continue
            
            # Skip if it contains any board word
            if any(board_word in clue or clue in board_word for board_word in self.words_on_board):
                continue
            
            # Calculate danger score for avoid words
            avoid_sims = [self._word_similarity(clue, w) for w in avoid_words if w]
            assassin_sim = (self._word_similarity(clue, self.assassin_word) 
                           if self.assassin_word else 0.0)
            
            # Skip dangerous clues
            if assassin_sim > 0.35:
                continue
            if any(sim > 0.4 for sim in avoid_sims):
                continue
            
            # Calculate connection strength to target words
            connected_words = [w for w, sim in data['similarities'].items() if sim > 0.3]
            if not connected_words:
                continue
            
            # Initial score calculation
            avg_sim = sum(data['similarities'].values()) / len(data['similarities'])
            max_sim = max(data['similarities'].values()) if data['similarities'] else 0
            
            # Adjust for word frequency/familiarity
            frequency_bonus = self.word_frequency.get(clue, 0.5) * 0.3
            
            # Prefer moderate-length words (not too short, not too long)
            length_bonus = 0.2 if 4 <= len(clue) <= 8 else 0
            
            # Calculate final score
            data['score'] = (
                avg_sim * 0.5 + 
                max_sim * 0.5 + 
                frequency_bonus +
                length_bonus - 
                assassin_sim * 2.0 -
                (sum(avoid_sims) / len(avoid_sims) if avoid_sims else 0) * 1.0
            )
            
            # Count connected targets
            data['connected'] = len(connected_words)
            data['words'] = connected_words
            
            # Only add if it connects to at least one word
            if data['connected'] > 0:
                filtered_candidates[clue] = data
        
        # Convert to a sorted list
        scored_candidates = [(clue, data['score'], data['connected'], data['words']) 
                            for clue, data in filtered_candidates.items()]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return top candidates
        return scored_candidates[:max_candidates]

    def _simulate_guess(self, clue, remaining_words):

        # Calculate similarity of all words to the clue
        similarities = {}
        for word in remaining_words:
            similarities[word] = self._word_similarity(clue, word)
        
        # Sort words by similarity to clue
        sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Determine guessed words based on similarity
        guesses = []
        for word, sim in sorted_words:
            # Skip words with very low similarity
            if sim < 0.2:
                continue
                
            # Add this guess
            guesses.append(word)
            
            # Determine outcome based on the guessed word
            if word == self.assassin_word:
                return guesses, 'assassin'
            elif word in self.opponent_words:
                return guesses, 'opponent'
            elif word in self.civilian_words:
                return guesses, 'civilian'
            
            # If all target words are guessed, it's a win
            if all(w in guesses for w in self.my_words):
                return guesses, 'win'
        
        # If we get here, the turn continues or ends
        return guesses, 'continue'

    class MCTSNode:
        """
        Node in the MCTS tree representing a game state after a clue.
        """
        def __init__(self, clue, num, parent=None):
            self.clue = clue          # The clue word
            self.num = num            # Number of words intended
            self.parent = parent      # Parent node
            self.children = []        # Child nodes
            self.visits = 0           # Number of visits
            self.value = 0.0          # Total value from simulations
            self.untried_actions = [] # Actions not yet explored
        
        def uct_value(self, exploration_weight, parent_visits):
            """Calculate UCT value for node selection"""
            if self.visits == 0:
                return float('inf')
            
            exploitation = self.value / self.visits
            exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
            return exploitation + exploration
        
        def best_child(self, exploration_weight):
            """Select best child based on UCT value"""
            if not self.children:
                return None
                
            return max(
                self.children,
                key=lambda c: c.uct_value(exploration_weight, self.visits)
            )
        
        def expand(self, action):
            """Add a new child node for the given action"""
            clue, num = action
            child = CodemasterMCTS.MCTSNode(clue, num, self)
            self.children.append(child)
            return child
        
        def is_fully_expanded(self):
            """Check if all actions have been tried"""
            return not self.untried_actions

    def _get_mcts_actions(self, state):
        """
        Get possible actions (clues) for MCTS.
        """
        target_words = state.get('target_words', [])
        avoid_words = state.get('avoid_words', [])
        
        # Get candidate clues
        candidates = self._find_potential_clues(target_words, avoid_words)
        
        # Convert to actions (clue, num)
        actions = []
        for clue, _, connected, words in candidates:
            # Determine the appropriate number for the clue
            if connected > 1:
                # For multi-word clues, use the actual number of connected words
                num = min(connected, len(target_words))
            else:
                # For single-word clues, just use 1
                num = 1
                
            actions.append((clue, num))
        
        return actions

    def _mcts_simulate(self, state, clue, num):
        """
        Run a single simulation from the given state using the clue.
        Returns the reward from this simulation.
        """
        # Copy state for simulation
        sim_state = {
            'target_words': state['target_words'].copy(),
            'avoid_words': state['avoid_words'].copy(),
            'remaining_words': set(self.words_on_board).copy()
        }
        
        # Simulate the guesser's behavior
        guesses, outcome = self._simulate_guess(clue, list(sim_state['remaining_words']))
        
        # Calculate reward based on outcome
        reward = 0.0
        
        # Count correctly guessed target words
        correct_guesses = [w for w in guesses if w in sim_state['target_words']]
        reward += len(correct_guesses) * self.target_reward
        
        # Handle different outcomes
        if outcome == 'win':
            reward += self.win_reward
        elif outcome == 'assassin':
            reward += self.assassin_penalty
        elif outcome == 'opponent':
            reward += self.opponent_penalty
        elif outcome == 'civilian':
            reward += self.civilian_penalty
        
        # Adjust reward based on clue efficiency
        efficiency = len(correct_guesses) / max(1, num)
        reward *= (0.5 + 0.5 * efficiency)
        
        # Small bonus for multi-word clues
        if num > 1:
            reward *= (1.0 + 0.1 * (num - 1))
        
        return reward

    def _run_mcts(self, max_iterations=None):
        """
        Run MCTS to find the best clue.
        Returns (clue, num) for the best action found.
        """
        if not self.my_words:
            return "PASS", 0
            
        # Initial game state
        initial_state = {
            'target_words': self.my_words.copy(),
            'avoid_words': self.opponent_words + ([self.assassin_word] if self.assassin_word else []),
            'remaining_words': set(self.words_on_board).copy()
        }
        
        # Create root node
        root = self.MCTSNode(None, 0)
        root.untried_actions = self._get_mcts_actions(initial_state)
        
        # Set iterations
        iterations = max_iterations or self.num_simulations
        
        # MCTS main loop
        for _ in range(iterations):
            # Selection phase
            node = root
            state = initial_state.copy()
            
            # Select until we reach a leaf node
            while node.is_fully_expanded() and node.children:
                node = node.best_child(self.exploration_weight)
                
                # Update state (not needed here since we're just choosing clues)
                
            # Expansion phase
            if not node.is_fully_expanded() and node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                
                # Create a new child node
                node = node.expand(action)
            
            # Simulation phase
            clue, num = node.clue, node.num
            reward = self._mcts_simulate(state, clue, num)
            
            # Backpropagation phase
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent
        
        # Select best child of root as our answer
        best_node = None
        best_value = float('-inf')
        
        for child in root.children:
            # Skip nodes with no visits
            if child.visits == 0:
                continue
                
            # Calculate average value
            avg_value = child.value / child.visits
            
            # Add a bonus for multi-word clues
            bonus = 0.1 * (child.num - 1) if child.num > 1 else 0
            score = avg_value + bonus
            
            if score > best_value:
                best_value = score
                best_node = child
        
        # If we found a good node, return its clue and num
        if best_node:
            return best_node.clue, best_node.num
        
        # Fallback to first action if no node visited
        if root.untried_actions:
            return root.untried_actions[0]
        
        # Last resort
        return "PASS", 0

    def get_clue(self):
        """
        Use true MCTS to find the best clue for the current board state.
        """
        # Run MCTS to find best clue
        clue, num = self._run_mcts(self.num_simulations)
        
        return clue.upper(), num