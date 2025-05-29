from players.codemaster import Codemaster
import random
import numpy as np
from model_manager import get_glove_model, get_sbert_model

class CodemasterSBERT(Codemaster):
    """
    Codemaster that uses Sentence Transformers (SBERT) for semantic similarity
    but uses GloVe vocabulary for candidate clues (not hardcoded)
    """
    
    def __init__(self, team="Red", sbert_model="all-MiniLM-L6-v2", model_name="glove-wiki-gigaword-300"):
        super().__init__()
        self.team = team
        # Use shared models
        print(f"CodemasterSBERT: Using shared SBERT model '{sbert_model}'...")
        self.sbert_model = get_sbert_model(sbert_model)
        
        # Use shared model instead of loading new instance
        print(f"Using shared {model_name} model...")
        self.glove_model = get_glove_model(model_name)
        
        # Game state
        self.words_on_board = []
        self.key_grid = []
        self.my_words = []
        self.opponent_words = []
        self.civilian_words = []
        self.assassin_word = None
        
        # Create vocabulary from GloVe (not hardcoded!)
        vocab_items = sorted(self.glove_model.key_to_index.items(), key=lambda kv: kv[1])[:15000]
        self.vocabulary = [w for w, _ in vocab_items if w.isalpha() and len(w) >= 3]
        
        # Word frequency for quality scoring
        self.word_frequency = {}
        for i, (word, _) in enumerate(vocab_items):
            self.word_frequency[word] = 1.0 - (i / len(vocab_items))
    
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
    
    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    def _get_sbert_similarity(self, word1, word2):
        """Get SBERT similarity between two words"""
        try:
            embeddings = self.sbert_model.encode([word1, word2])
            return self._cosine_similarity(embeddings[0], embeddings[1])
        except:
            return 0.0
    
# Replace the _find_clues_for_words method in CodemasterSBERT with this faster version

    def _find_clues_for_words(self, target_words, avoid_words, num_candidates=50):  # Reduced from 100
        """Find potential clues using SBERT similarity with GloVe vocabulary (FASTER VERSION)"""
        if not target_words:
            return []
        
        clue_scores = {}
        board_words_set = set(w.lower() for w in self.words_on_board)
        
        # Get candidate clues from GloVe vocabulary (MUCH SMALLER SET)
        candidate_clues = []
        for clue in self.vocabulary[:2000]:  # Only check first 2000 words, not 15000!
            # Skip board words and derived words
            if (clue not in board_words_set and
                not any(board_word in clue or clue in board_word 
                    for board_word in self.words_on_board)):
                candidate_clues.append(clue)
        
        # Further limit for speed - only test top 500 candidates
        candidate_clues = candidate_clues[:500]
        
        print(f"Evaluating {len(candidate_clues)} candidate clues...")  # Debug info
        
        # Pre-encode all words for efficiency (batch encoding is faster)
        all_words = target_words + [w for w in avoid_words if w] + candidate_clues
        if self.assassin_word:
            all_words.append(self.assassin_word)
        
        try:
            # Batch encode all words at once (more efficient than one-by-one)
            all_embeddings = self.sbert_model.encode(all_words, show_progress_bar=False)
            word_to_embedding = {word: embedding for word, embedding in zip(all_words, all_embeddings)}
        except Exception as e:
            print(f"Batch encoding failed: {e}, falling back to individual encoding")
            word_to_embedding = {}
            for word in all_words:
                try:
                    word_to_embedding[word] = self.sbert_model.encode(word)
                except:
                    continue
        
        # Score each candidate clue using pre-computed embeddings
        for clue in candidate_clues:
            if clue not in word_to_embedding:
                continue
                
            clue_embedding = word_to_embedding[clue]
            target_similarities = []
            avoid_similarities = []
            
            # Calculate similarities to target words
            for target in target_words:
                if target in word_to_embedding:
                    target_embedding = word_to_embedding[target]
                    sim = self._cosine_similarity(clue_embedding, target_embedding)
                    target_similarities.append(sim)
            
            # Calculate similarities to avoid words
            for avoid_word in avoid_words:
                if avoid_word and avoid_word in word_to_embedding:
                    avoid_embedding = word_to_embedding[avoid_word]
                    sim = self._cosine_similarity(clue_embedding, avoid_embedding)
                    avoid_similarities.append(sim)
            
            # Calculate metrics
            if not target_similarities:
                continue
                
            avg_target_sim = np.mean(target_similarities)
            max_target_sim = max(target_similarities)
            min_target_sim = min(target_similarities)
            connected_targets = sum(1 for sim in target_similarities if sim > 0.4)
            
            avg_avoid_sim = np.mean(avoid_similarities) if avoid_similarities else 0
            max_avoid_sim = max(avoid_similarities) if avoid_similarities else 0
            
            # Special check for assassin
            assassin_sim = 0
            if self.assassin_word and self.assassin_word in word_to_embedding:
                assassin_embedding = word_to_embedding[self.assassin_word]
                assassin_sim = self._cosine_similarity(clue_embedding, assassin_embedding)
            
            # Skip very dangerous clues
            if assassin_sim > 0.45 or max_avoid_sim > 0.5:
                continue
            
            # Calculate final score (same as before)
            frequency_bonus = self.word_frequency.get(clue, 0.5) * 0.2
            length_bonus = 0.1 if 4 <= len(clue) <= 8 else 0
            
            score = (avg_target_sim * 0.5 + 
                    max_target_sim * 0.3 + 
                    min_target_sim * 0.2 + 
                    frequency_bonus + 
                    length_bonus - 
                    avg_avoid_sim * 1.5 - 
                    assassin_sim * 3.0)
            
            # Only consider clues that connect to targets
            if connected_targets > 0 or (len(target_words) == 1 and max_target_sim > 0.3):
                clue_scores[clue] = {
                    'score': score,
                    'connected': connected_targets,
                    'target_sims': target_similarities,
                    'avoid_sims': avoid_similarities
                }
        
        # Sort and return top candidates
        scored_clues = [(clue, data['score'], data['connected'], data['target_sims']) 
                    for clue, data in clue_scores.items()]
        scored_clues.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(scored_clues)} valid clues")  # Debug info
        return scored_clues[:num_candidates]
    
    def get_clue(self):
        """Generate the best clue using SBERT similarity"""
        if not self.my_words:
            return "PASS", 0
        
        # Get remaining words
        remaining_words = [w for w in self.my_words if not w.startswith("*")]
        if not remaining_words:
            return "PASS", 0
        
        avoid_words = (self.opponent_words + self.civilian_words + 
                      ([self.assassin_word] if self.assassin_word else []))
        
        best_clue = None
        best_score = -float('inf')
        best_num = 1
        
        # Try single words
        for target in remaining_words:
            clues = self._find_clues_for_words([target], avoid_words, 20)
            for clue, score, connected, target_sims in clues:
                if score > best_score:
                    best_clue = clue
                    best_score = score
                    best_num = 1
        
        # Try pairs of words
        if len(remaining_words) >= 2:
            for i in range(len(remaining_words)):
                for j in range(i + 1, min(i + 3, len(remaining_words))):
                    target_pair = [remaining_words[i], remaining_words[j]]
                    clues = self._find_clues_for_words(target_pair, avoid_words, 15)
                    
                    for clue, score, connected, target_sims in clues:
                        # Bonus for multi-word clues
                        multi_bonus = 0.8 if connected >= 2 else 0
                        adjusted_score = score + multi_bonus
                        
                        if adjusted_score > best_score and connected >= 2:
                            best_clue = clue
                            best_score = adjusted_score
                            best_num = 2
        
        # Try three words
        if len(remaining_words) >= 3:
            for i in range(min(2, len(remaining_words) - 2)):
                target_triple = remaining_words[i:i+3]
                clues = self._find_clues_for_words(target_triple, avoid_words, 10)
                
                for clue, score, connected, target_sims in clues:
                    multi_bonus = 0.5 if connected >= 3 else 0.3 if connected >= 2 else 0
                    adjusted_score = score + multi_bonus
                    
                    if adjusted_score > best_score and connected >= 2:
                        best_clue = clue
                        best_score = adjusted_score
                        best_num = min(connected, 3)
        
        # Fallback if no good clue found
        if not best_clue or best_score < -0.5:
            if remaining_words:
                safe_clues = self._find_clues_for_words([remaining_words[0]], avoid_words, 5)
                if safe_clues:
                    best_clue = safe_clues[0][0]
                    best_num = 1
                else:
                    best_clue = "CONCEPT"
                    best_num = 1
        
        return str(best_clue).upper(), best_num