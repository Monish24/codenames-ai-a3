from players.codemaster import Codemaster
import random
import numpy as np
import unicodedata
import re
from model_manager import get_glove_model

class CodemasterEmbeddings(Codemaster):
    """
    Codemaster that uses word embeddings (GloVe) to find semantic clues
    Enhanced with proper validation and anti-cheating measures
    """
    
    def __init__(self, team="Red", model_name="glove-wiki-gigaword-300"):
        super().__init__()
        self.team = team
        # Use shared model instead of loading new instance
        print(f"Using shared {model_name} model...")
        self.model = get_glove_model(model_name)
        
        # Game state
        self.words_on_board = []
        self.key_grid = []
        self.my_words = []
        self.opponent_words = []
        self.civilian_words = []
        self.assassin_word = None
        
        # Create vocabulary for potential clues - improved filtering
        vocab_items = sorted(self.model.key_to_index.items(), key=lambda kv: kv[1])[:10000]
        self.vocabulary = [w for w, _ in vocab_items if self._is_valid_vocabulary_word(w)]
        
        # Word frequency for quality scoring
        self.word_frequency = {}
        for i, (word, _) in enumerate(vocab_items):
            self.word_frequency[word] = 1.0 - (i / len(vocab_items))
        
    def _is_valid_vocabulary_word(self, word):
        """Check if word is suitable for vocabulary"""
        # Must be alphabetic and reasonable length
        if not word.isalpha() or len(word) < 3 or len(word) > 12:
            return False
        
        # No words with repeated characters (often typos)
        if len(set(word)) < len(word) * 0.6:  # At least 60% unique characters
            return False
        
        # Basic proper noun filter (very basic - just check if starts with capital)
        if word[0].isupper() and len(word) > 4:
            return False
            
        return True
    
    def _normalize_word(self, word):
        """Remove accents and normalize word for comparison"""
        word = word.lower()
        # Remove accents and diacritical marks
        normalized = unicodedata.normalize('NFD', word)
        ascii_word = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        return ascii_word
    
    def _is_invalid_clue(self, clue, board_words):
        """Enhanced validation to prevent cheating and ensure quality"""
        if not clue:
            return True
            
        clue_normalized = self._normalize_word(clue)
        
        # Check against board words (normalized to prevent accent cheating)
        for board_word in board_words:
            board_normalized = self._normalize_word(board_word)
            
            # Exact match
            if clue_normalized == board_normalized:
                return True
            
            # Substring match (either direction)
            if clue_normalized in board_normalized or board_normalized in clue_normalized:
                return True
            
            # Similar word check (prevent slight variations)
            if len(clue_normalized) > 3 and len(board_normalized) > 3:
                # Check if words are very similar (Levenshtein-like)
                if self._words_too_similar(clue_normalized, board_normalized):
                    return True
        
        return False
    
    def _words_too_similar(self, word1, word2):
        """Check if two words are too similar (simple similarity check)"""
        if len(word1) != len(word2):
            return False
        
        # Count differing characters
        diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
        
        # Too similar if only 1-2 characters different
        return diff_count <= 2 and len(word1) > 4
    
    def _validate_clue_quality(self, clue):
        """Ensure clue meets quality standards"""
        if not clue:
            return False
            
        # Must be pure alphabetic (no accents, numbers, punctuation)
        if not clue.isalpha():
            return False
        
        # Reasonable length
        if len(clue) < 3 or len(clue) > 15:
            return False
        
        # Must be in our model's vocabulary
        if clue.lower() not in self.model.key_to_index:
            return False
        
        # Check word frequency - avoid very rare words
        freq = self.word_frequency.get(clue.lower(), 0)
        if freq < 0.0001:  # Very rare words
            return False
        
        # Avoid words with unusual patterns (potential typos/foreign words)
        if re.search(r'[qxz]{2,}|[aeiou]{4,}|[bcdfghjklmnpqrstvwxyz]{4,}', clue.lower()):
            return False
        
        return True
    
    def set_game_state(self, words_on_board, key_grid):
        """Set the current game state"""
        self.words_on_board = [w.lower() for w in words_on_board]
        self.key_grid = key_grid
        
        # Determine team color
        self.color = self.team or next(
            (k for k in key_grid if k not in ("Civilian", "Assassin")), "Red"
        )
        
        # Categorize words by team
        self.my_words = []
        self.opponent_words = []
        self.civilian_words = []
        self.assassin_word = None
        
        for word, key in zip(self.words_on_board, self.key_grid):
            if key == self.color:
                self.my_words.append(word)
            elif key == "Assassin":
                self.assassin_word = word
            elif key == "Civilian":
                self.civilian_words.append(word)
            else:
                self.opponent_words.append(word)
    
    def _word_similarity(self, w1, w2):
        """Calculate cosine similarity between two words"""
        w1, w2 = w1.lower(), w2.lower()
        if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
            return 0.0
        v1 = self.model[w1]
        v2 = self.model[w2]
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def _find_clues_for_words(self, target_words, avoid_words, num_candidates=100):
        """Find potential clues for target words while avoiding dangerous words"""
        if not target_words:
            return []
            
        clue_scores = {}
        
        # Get candidate clues from similar words
        for target in target_words:
            if target not in self.model.key_to_index:
                continue
                
            try:
                # Get words similar to this target
                similar_words = self.model.most_similar(positive=[target], topn=50)
                for clue, _ in similar_words:
                    # Enhanced validation for each candidate
                    if (self._validate_clue_quality(clue) and 
                        not self._is_invalid_clue(clue, self.words_on_board)):
                        
                        if clue not in clue_scores:
                            clue_scores[clue] = {'target_sims': [], 'avoid_sims': []}
                        
                        clue_scores[clue]['target_sims'].append(self._word_similarity(clue, target))
            except:
                continue
        
        # Calculate avoid similarities
        for clue in clue_scores:
            for avoid_word in avoid_words:
                if avoid_word:
                    clue_scores[clue]['avoid_sims'].append(self._word_similarity(clue, avoid_word))
        
        # Score each clue with improved metrics
        scored_clues = []
        for clue, data in clue_scores.items():
            if not data['target_sims']:
                continue
                
            # Calculate metrics
            avg_target_sim = np.mean(data['target_sims'])
            max_target_sim = max(data['target_sims'])
            min_target_sim = min(data['target_sims'])
            connected_targets = sum(1 for sim in data['target_sims'] if sim > 0.3)
            
            avg_avoid_sim = np.mean(data['avoid_sims']) if data['avoid_sims'] else 0
            max_avoid_sim = max(data['avoid_sims']) if data['avoid_sims'] else 0
            
            # Enhanced assassin check
            assassin_sim = 0
            if self.assassin_word:
                assassin_sim = self._word_similarity(clue, self.assassin_word)
            
            # Skip dangerous clues with stricter thresholds
            if max_avoid_sim > 0.35 or assassin_sim > 0.3:
                continue
            
            # Enhanced scoring with frequency bonus
            frequency_bonus = self.word_frequency.get(clue.lower(), 0.5) * 0.2
            length_bonus = 0.1 if 4 <= len(clue) <= 8 else 0  # Prefer moderate length
            
            # Calculate final score
            score = (avg_target_sim * 0.4 + 
                    max_target_sim * 0.3 + 
                    min_target_sim * 0.1 +  # Ensures all targets are reasonably connected
                    frequency_bonus + 
                    length_bonus - 
                    avg_avoid_sim * 2.0 - 
                    assassin_sim * 3.0)
            
            scored_clues.append((clue, score, connected_targets, data['target_sims']))
        
        # Sort by score and return top candidates
        scored_clues.sort(key=lambda x: x[1], reverse=True)
        return scored_clues[:num_candidates]
    
    def get_clue(self):
        """Generate the best clue for current game state"""
        if not self.my_words:
            return "PASS", 0
        
        avoid_words = (self.opponent_words + self.civilian_words + 
                      ([self.assassin_word] if self.assassin_word else []))
        
        best_clue = None
        best_score = -float('inf')
        best_num = 1
        
        # Try different combinations of target words
        remaining_words = [w for w in self.my_words if not w.startswith("*")]
        
        if not remaining_words:
            return "PASS", 0
        
        # Try single words first
        for target in remaining_words:
            clues = self._find_clues_for_words([target], avoid_words, 25)
            for clue, score, connected, target_sims in clues:
                if score > best_score:
                    best_clue = clue
                    best_score = score
                    best_num = 1
        
        # Try pairs of words with increased bonus
        if len(remaining_words) >= 2:
            for i in range(len(remaining_words)):
                for j in range(i + 1, min(i + 4, len(remaining_words))):  # Try more combinations
                    target_pair = [remaining_words[i], remaining_words[j]]
                    clues = self._find_clues_for_words(target_pair, avoid_words, 20)
                    
                    for clue, score, connected, target_sims in clues:
                        # Increased bonus for multi-word clues
                        multi_bonus = 0.5 if connected >= 2 else 0  # Increased from 0.2
                        adjusted_score = score + multi_bonus
                        
                        if adjusted_score > best_score and connected >= 2:
                            best_clue = clue
                            best_score = adjusted_score
                            best_num = 2
        
        # Try three words if we have enough
        if len(remaining_words) >= 3:
            for i in range(min(3, len(remaining_words) - 2)):
                target_triple = remaining_words[i:i+3]
                clues = self._find_clues_for_words(target_triple, avoid_words, 15)
                
                for clue, score, connected, target_sims in clues:
                    # Big bonus for triplet connections
                    multi_bonus = 1.0 if connected >= 3 else 0.5 if connected >= 2 else 0  # Increased
                    adjusted_score = score + multi_bonus
                    
                    if adjusted_score > best_score and connected >= 2:
                        best_clue = clue
                        best_score = adjusted_score
                        best_num = min(connected, 3)
        
        # Enhanced fallback with better validation
        if not best_clue or best_score < -0.3:  # More lenient threshold
            # Try to find any safe single-word clue
            fallback_candidates = []
            for target in remaining_words:
                simple_clues = self._find_clues_for_words([target], avoid_words, 10)
                fallback_candidates.extend(simple_clues)
            
            # Sort fallback candidates and pick the best safe one
            if fallback_candidates:
                fallback_candidates.sort(key=lambda x: x[1], reverse=True)
                for clue, score, connected, target_sims in fallback_candidates:
                    if score > -0.5:  # Must be reasonably good
                        best_clue = clue
                        best_num = 1
                        break
            
            # Ultimate fallback - use a safe generic word
            if not best_clue:
                safe_words = ["WORD", "THING", "ITEM", "OBJECT", "CONCEPT"]
                for safe_word in safe_words:
                    if not self._is_invalid_clue(safe_word, self.words_on_board):
                        best_clue = safe_word
                        best_num = 1
                        break
                
                # If even safe words fail, use HINT
                if not best_clue:
                    best_clue = "HINT"
                    best_num = 1
        
        return str(best_clue).upper(), best_num