from players.codemaster import Codemaster
import random
import numpy as np
from model_manager import get_glove_model
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import itertools


class CodemasterCurriculum(Codemaster):
    
    def __init__(self, team="Red", model_name="glove-wiki-gigaword-300"):
        super().__init__()
        self.team = team
        print(f"Using shared {model_name} model...")
        self.model = get_glove_model(model_name)
        
        # Game state
        self.words_on_board = []
        self.key_grid = []
        self.my_words = []
        self.opponent_words = []
        self.civilian_words = []
        self.assassin_word = None
        self.used_clues = set()
        
        self.difficulty_level = 1  # Start with easiest level
        self.success_history = deque(maxlen=20)  # Track recent successes
        self.concept_success_rates = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self.learning_progress = {'mastered_levels': set(), 'current_focus': 1}
        
        # Dynamic vocabulary with quality tiers
        self.vocabulary_tiers = self._create_vocabulary_tiers()
        
        # Concept discovery cache to avoid recomputation
        self.concept_cache = {}
        
        # Progressive difficulty thresholds
        self.difficulty_thresholds = {
            1: {'min_similarity': 0.5, 'max_targets': 2, 'safety_margin': 0.4},
            2: {'min_similarity': 0.4, 'max_targets': 3, 'safety_margin': 0.35},
            3: {'min_similarity': 0.3, 'max_targets': 4, 'safety_margin': 0.3},
            4: {'min_similarity': 0.25, 'max_targets': 5, 'safety_margin': 0.25}
        }
    
    def _create_vocabulary_tiers(self):
        """Create vocabulary tiers based on frequency and conceptual utility"""
        vocab_items = sorted(self.model.key_to_index.items(), key=lambda kv: kv[1])
        
        tiers = {
            'basic': [],      # Top 1000 - everyday concepts
            'intermediate': [], # 1000-3000 - educated vocabulary
            'advanced': [],   # 3000-6000 - sophisticated concepts
            'expert': []      # 6000-10000 - specialized terms
        }
        
        for i, (word, _) in enumerate(vocab_items[:10000]):
            if self._is_valid_concept_word(word):
                if i < 1000:
                    tiers['basic'].append(word)
                elif i < 3000:
                    tiers['intermediate'].append(word)
                elif i < 6000:
                    tiers['advanced'].append(word)
                else:
                    tiers['expert'].append(word)
        
        return tiers
    
    def _is_valid_concept_word(self, word):
        """Enhanced validation for concept words"""
        if not word.isalpha() or len(word) < 3 or len(word) > 15:
            return False
        
        # Filter likely proper nouns
        if word[0].isupper() and len(word) > 4:
            return False
        
        # Prefer words with good conceptual patterns
        conceptual_endings = ['ing', 'tion', 'ness', 'ment', 'able', 'ful', 'ity', 'ism']
        has_conceptual_ending = any(word.endswith(ending) for ending in conceptual_endings)
        
        # Allow shorter common words or words with conceptual endings
        if len(word) <= 6 or has_conceptual_ending:
            return True
        
        # Filter very long words unless they look conceptual
        return len(word) <= 10 and has_conceptual_ending
    
    def set_game_state(self, words_on_board, key_grid):
        """Enhanced game state setting with adaptation"""
        self.words_on_board = [w.lower() for w in words_on_board]
        self.key_grid = key_grid
        
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
        
        # Adapt difficulty based on current game state
        self._adapt_difficulty_to_game_state()
    
    def _adapt_difficulty_to_game_state(self):
        """Dynamically adapt difficulty based on game progress"""
        remaining_words = [w for w in self.my_words if not w.startswith("*")]
        total_my_words = len(self.my_words)
        
        if not remaining_words or total_my_words == 0:
            return
        
        # Calculate game progress
        progress = 1.0 - (len(remaining_words) / total_my_words)
        
        # Increase difficulty as game progresses (fewer words = higher risk tolerance)
        if progress > 0.7:  # Late game - be more aggressive
            self.difficulty_level = min(4, self.difficulty_level + 1)
        elif progress > 0.4:  # Mid game
            self.difficulty_level = min(3, max(2, self.difficulty_level))
        else:  # Early game - be conservative
            self.difficulty_level = max(1, self.difficulty_level - 1)
        
        # Also consider success history
        if len(self.success_history) >= 5:
            recent_success_rate = sum(self.success_history[-5:]) / 5
            if recent_success_rate > 0.7:
                self.difficulty_level = min(4, self.difficulty_level + 1)
            elif recent_success_rate < 0.3:
                self.difficulty_level = max(1, self.difficulty_level - 1)
    
    def _safe_word_similarity(self, w1, w2, default=0.0):
        """Safe similarity calculation with enhanced error handling"""
        try:
            w1, w2 = w1.lower().strip(), w2.lower().strip()
            
            if not w1 or not w2 or w1 == w2:
                return default
                
            if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
                return default
                
            v1 = self.model[w1]
            v2 = self.model[w2]
            
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return default
                
            similarity = float(np.dot(v1, v2) / (norm1 * norm2))
            return max(-1.0, min(1.0, similarity))
            
        except Exception:
            return default
    
    def _discover_concept_clusters(self, target_words):
        """
        Enhanced curriculum learning with progressive difficulty levels
        """
        if not target_words:
            return []
        
        # Create cache key
        cache_key = tuple(sorted(target_words))
        if cache_key in self.concept_cache:
            return self.concept_cache[cache_key]
        
        # Get vectors for target words
        target_vectors = []
        valid_targets = []
        for word in target_words:
            if word in self.model.key_to_index:
                target_vectors.append(self.model[word])
                valid_targets.append(word)
        
        if not valid_targets:
            return []
        
        concepts = []
        
        # Progressive discovery based on difficulty level
        if self.difficulty_level >= 1:
            concepts.extend(self._find_direct_similarity_concepts(valid_targets))
        
        if self.difficulty_level >= 2:
            concepts.extend(self._find_categorical_concepts(valid_targets, target_vectors))
        
        if self.difficulty_level >= 3:
            concepts.extend(self._find_abstract_concepts(valid_targets, target_vectors))
        
        if self.difficulty_level >= 4:
            concepts.extend(self._find_expert_concepts(valid_targets, target_vectors))
        
        # Cache results
        self.concept_cache[cache_key] = concepts
        
        return concepts
    
    def _find_direct_similarity_concepts(self, target_words):
        """Level 1: Direct similarity concepts (curriculum foundation)"""
        concepts = []
        vocabulary = self.vocabulary_tiers['basic'] + self.vocabulary_tiers['intermediate'][:500]
        
        # For each target, find highly similar words
        neighbor_scores = defaultdict(lambda: {'targets': [], 'similarities': [], 'frequency_bonus': 0})
        
        for target in target_words:
            try:
                similar_words = self.model.most_similar(positive=[target], topn=25)
                for neighbor, sim in similar_words:
                    if self._is_valid_clue_candidate(neighbor):
                        neighbor_scores[neighbor]['targets'].append(target)
                        neighbor_scores[neighbor]['similarities'].append(sim)
                        
                        # Frequency bonus for common words
                        if neighbor in vocabulary:
                            neighbor_scores[neighbor]['frequency_bonus'] = 0.1
            except:
                continue
        
        # Create concepts from neighbors that connect multiple targets
        for neighbor, data in neighbor_scores.items():
            if len(data['targets']) >= 1:
                avg_sim = np.mean(data['similarities'])
                connection_bonus = 0.8 if len(data['targets']) > 1 else 0
                
                score = avg_sim + connection_bonus + data['frequency_bonus']
                
                concepts.append({
                    'clue': neighbor,
                    'targets': data['targets'],
                    'score': score,
                    'level': 1,
                    'type': 'direct_similarity',
                    'confidence': avg_sim
                })
        
        concepts.sort(key=lambda x: x['score'], reverse=True)
        return concepts[:8]  # Top 8 direct concepts
    
    def _find_categorical_concepts(self, target_words, target_vectors):
        """Level 2: Categorical/hypernym concepts"""
        concepts = []
        
        if len(target_vectors) < 2:
            return concepts
        
        # Calculate semantic centroid
        centroid = np.mean(target_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        vocabulary = self.vocabulary_tiers['intermediate'] + self.vocabulary_tiers['advanced'][:300]
        
        for word in vocabulary:
            if (word in self.model.key_to_index and 
                self._is_valid_clue_candidate(word)):
                
                word_vector = self.model[word]
                word_vector = word_vector / np.linalg.norm(word_vector)
                
                # Check if word could be a category
                centroid_sim = np.dot(centroid, word_vector)
                
                # Calculate individual target similarities
                individual_sims = [self._safe_word_similarity(word, target) 
                                 for target in target_words]
                
                # Good categories: moderate similarity to all, consistent connections
                if individual_sims and all(0.2 <= sim <= 0.5 for sim in individual_sims):
                    avg_sim = np.mean(individual_sims)
                    consistency = 1.0 - np.var(individual_sims)  # Reward consistency
                    
                    # Check if word looks like a category
                    category_likelihood = self._assess_category_likelihood(word)
                    
                    if category_likelihood > 0.4:
                        score = (centroid_sim * 0.4 + avg_sim * 0.3 + 
                                consistency * 0.2 + category_likelihood * 0.1)
                        
                        concepts.append({
                            'clue': word,
                            'targets': target_words,
                            'score': score,
                            'level': 2,
                            'type': 'categorical',
                            'confidence': consistency
                        })
        
        concepts.sort(key=lambda x: x['score'], reverse=True)
        return concepts[:6]
    
    def _find_abstract_concepts(self, target_words, target_vectors):
        """Level 3: Abstract/metaphorical concepts"""
        concepts = []
        
        if len(target_vectors) < 2:
            return concepts
        
        vocabulary = self.vocabulary_tiers['advanced']
        
        # Use more sophisticated analysis for abstract concepts
        centroid = np.mean(target_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        for word in vocabulary[:400]:  # Limit for performance
            if (word in self.model.key_to_index and 
                self._is_valid_clue_candidate(word)):
                
                # Check abstraction level
                abstraction_score = self._assess_abstraction_level(word)
                
                if abstraction_score > 0.5:  # Must be reasonably abstract
                    word_vector = self.model[word] / np.linalg.norm(self.model[word])
                    centroid_sim = np.dot(centroid, word_vector)
                    
                    # Abstract concepts should connect moderately to targets
                    individual_sims = [self._safe_word_similarity(word, target) 
                                     for target in target_words]
                    
                    if individual_sims and all(0.15 <= sim <= 0.4 for sim in individual_sims):
                        avg_sim = np.mean(individual_sims)
                        
                        score = (centroid_sim * 0.5 + avg_sim * 0.3 + abstraction_score * 0.2)
                        
                        concepts.append({
                            'clue': word,
                            'targets': target_words,
                            'score': score,
                            'level': 3,
                            'type': 'abstract',
                            'confidence': abstraction_score
                        })
        
        concepts.sort(key=lambda x: x['score'], reverse=True)
        return concepts[:4]
    
    def _find_expert_concepts(self, target_words, target_vectors):
        """Level 4: Expert-level sophisticated concepts"""
        concepts = []
        
        if len(target_vectors) < 3:  # Only for complex multi-word clues
            return concepts
        
        vocabulary = self.vocabulary_tiers['expert']
        
        # Expert level: find very sophisticated connections
        centroid = np.mean(target_vectors, axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        
        # Also try vector arithmetic for expert concepts
        expert_concepts = self._discover_vector_arithmetic_concepts(target_words, target_vectors)
        concepts.extend(expert_concepts)
        
        for word in vocabulary[:200]:
            if (word in self.model.key_to_index and 
                self._is_valid_clue_candidate(word)):
                
                sophistication = self._assess_concept_sophistication(word)
                
                if sophistication > 0.6:
                    word_vector = self.model[word] / np.linalg.norm(self.model[word])
                    centroid_sim = np.dot(centroid, word_vector)
                    
                    # Expert concepts can have more varied similarity patterns
                    individual_sims = [self._safe_word_similarity(word, target) 
                                     for target in target_words]
                    
                    if individual_sims and np.mean(individual_sims) > 0.2:
                        variance_penalty = np.var(individual_sims)
                        
                        score = (centroid_sim * 0.4 + np.mean(individual_sims) * 0.4 + 
                                sophistication * 0.2 - variance_penalty)
                        
                        concepts.append({
                            'clue': word,
                            'targets': target_words,
                            'score': score,
                            'level': 4,
                            'type': 'expert',
                            'confidence': sophistication
                        })
        
        concepts.sort(key=lambda x: x['score'], reverse=True)
        return concepts[:3]
    
    def _discover_vector_arithmetic_concepts(self, target_words, target_vectors):
        """Expert technique: use vector arithmetic to find relationships"""
        concepts = []
        
        if len(target_vectors) < 3:
            return concepts
        
        try:
            # Try different vector combinations
            for i in range(len(target_words)):
                for j in range(i + 1, len(target_words)):
                    # Calculate relationship vector
                    v1, v2 = target_vectors[i], target_vectors[j]
                    relationship = v2 - v1
                    relationship = relationship / np.linalg.norm(relationship)
                    
                    # Find words that capture this relationship
                    candidates = self.model.similar_by_vector(relationship, topn=15)
                    
                    for word, sim in candidates:
                        if (self._is_valid_clue_candidate(word) and 
                            sim > 0.3):
                            
                            # Verify this word actually relates to our targets
                            target_sims = [self._safe_word_similarity(word, target) 
                                         for target in target_words]
                            
                            if target_sims and np.mean(target_sims) > 0.25:
                                concepts.append({
                                    'clue': word,
                                    'targets': target_words,
                                    'score': sim * np.mean(target_sims),
                                    'level': 4,
                                    'type': 'vector_arithmetic',
                                    'confidence': sim
                                })
        except:
            pass
        
        return concepts
    
    def _assess_category_likelihood(self, word):
        """Assess if word is likely a category/hypernym"""
        score = 0.3
        
        # Category words often have certain patterns
        if any(word.endswith(end) for end in ['ry', 'ty', 'cy', 'sm', 'al', 'ic']):
            score += 0.3
        
        if 4 <= len(word) <= 8:
            score += 0.2
        
        # Check semantic neighbors for category patterns
        try:
            neighbors = self.model.most_similar(positive=[word], topn=5)
            category_neighbors = sum(1 for n, _ in neighbors 
                                   if any(n.endswith(end) for end in ['ing', 'tion', 'ness']))
            score += (category_neighbors / 5) * 0.2
        except:
            pass
        
        return min(1.0, score)
    
    def _assess_abstraction_level(self, word):
        """Assess how abstract/conceptual a word is"""
        score = 0.3
        
        abstract_patterns = ['ness', 'ity', 'ism', 'ence', 'ance', 'tion', 'ment']
        if any(word.endswith(pattern) for pattern in abstract_patterns):
            score += 0.4
        
        if 5 <= len(word) <= 12:
            score += 0.2
        
        # Abstract words are often high-frequency
        vocab_rank = self.model.key_to_index.get(word, 10000)
        if vocab_rank < 2000:  # High frequency
            score += 0.3
        
        return min(1.0, score)
    
    def _assess_concept_sophistication(self, word):
        """Assess how sophisticated/expert-level a concept is"""
        score = 0.2
        
        # Sophisticated words often have complex morphology
        sophisticated_patterns = ['ology', 'ography', 'ification', 'ization']
        if any(pattern in word for pattern in sophisticated_patterns):
            score += 0.5
        
        # Moderate frequency (not too common, not too rare)
        vocab_rank = self.model.key_to_index.get(word, 10000)
        if 2000 <= vocab_rank <= 6000:
            score += 0.3
        
        # Length suggests complexity
        if 7 <= len(word) <= 12:
            score += 0.2
        
        return min(1.0, score)
    
    def _is_valid_clue_candidate(self, word):
        """Enhanced validation for clue candidates"""
        if (not word or not word.isalpha() or len(word) < 3 or 
            word in self.words_on_board or word.lower() in self.used_clues):
            return False
        
        # Check for substring matches with board words
        for board_word in self.words_on_board:
            if (word.lower() in board_word.lower() or 
                board_word.lower() in word.lower()):
                return False
        
        return True
    
    def _validate_concept_safety(self, concept, avoid_words):
        """Enhanced safety validation with adaptive thresholds"""
        clue = concept['clue']
        thresholds = self.difficulty_thresholds[self.difficulty_level]
        
        # Check similarity to avoid words
        max_avoid_sim = 0
        for avoid_word in avoid_words:
            if avoid_word:
                sim = self._safe_word_similarity(clue, avoid_word)
                max_avoid_sim = max(max_avoid_sim, sim)
        
        if max_avoid_sim > thresholds['safety_margin']:
            return False
        
        # Check assassin similarity with strict threshold
        if self.assassin_word:
            assassin_sim = self._safe_word_similarity(clue, self.assassin_word)
            if assassin_sim > 0.3:  # Always strict for assassin
                return False
        
        return True
    
    def record_clue_outcome(self, concept_type, success):
        """Record outcome for curriculum learning adaptation"""
        self.success_history.append(1 if success else 0)
        self.concept_success_rates[concept_type]['attempts'] += 1
        if success:
            self.concept_success_rates[concept_type]['successes'] += 1
        
        # Adapt difficulty based on recent performance
        if len(self.success_history) >= 5:
            recent_success_rate = sum(self.success_history[-5:]) / 5
            
            if recent_success_rate > 0.8 and self.difficulty_level < 4:
                self.difficulty_level += 1
                print(f"Curriculum: Advanced to difficulty level {self.difficulty_level}")
            elif recent_success_rate < 0.3 and self.difficulty_level > 1:
                self.difficulty_level -= 1
                print(f"Curriculum: Reduced to difficulty level {self.difficulty_level}")
    
    def get_clue(self):
        """Enhanced clue generation with curriculum learning"""
        if not self.my_words:
            return "PASS", 0
        
        remaining_words = [w for w in self.my_words if not w.startswith("*")]
        if not remaining_words:
            return "PASS", 0
        
        avoid_words = (self.opponent_words + self.civilian_words + 
                      ([self.assassin_word] if self.assassin_word else []))
        
        best_concept = None
        best_score = -1
        
        thresholds = self.difficulty_thresholds[self.difficulty_level]
        max_targets = min(thresholds['max_targets'], len(remaining_words))
        
        print(f"Curriculum Level: {self.difficulty_level}, Max targets: {max_targets}")
        
        # Try combinations based on current difficulty level
        for target_count in range(1, max_targets + 1):
            if target_count == 1:
                # Single words
                for target in remaining_words:
                    concepts = self._discover_concept_clusters([target])
                    self._evaluate_concepts(concepts, avoid_words, best_concept, best_score)
            
            else:
                # Multi-word combinations
                max_combinations = min(10, len(list(itertools.combinations(remaining_words, target_count))))
                
                for combination in itertools.combinations(remaining_words, target_count):
                    if max_combinations <= 0:
                        break
                    max_combinations -= 1
                    
                    concepts = self._discover_concept_clusters(list(combination))
                    best_concept, best_score = self._evaluate_concepts(
                        concepts, avoid_words, best_concept, best_score
                    )
        
        # Return best concept
        if best_concept and best_score > 0.3:  # Minimum quality threshold
            clue = best_concept['clue'].upper()
            num_targets = len(best_concept['targets'])
            
            self.used_clues.add(clue.lower())
            
            print(f"Curriculum: Selected {best_concept['type']} concept (Level {best_concept['level']})")
            
            return clue, num_targets
        
        # Fallback
        return "LEARNING", 1
    
    def _evaluate_concepts(self, concepts, avoid_words, current_best, current_best_score):
        """Evaluate and update best concept"""
        best_concept = current_best
        best_score = current_best_score
        
        for concept in concepts:
            if self._validate_concept_safety(concept, avoid_words):
                # Apply curriculum bonuses
                level_bonus = (concept['level'] - 1) * 0.1  # Reward higher-level thinking
                confidence_bonus = concept.get('confidence', 0.5) * 0.2
                
                adjusted_score = concept['score'] + level_bonus + confidence_bonus
                
                if adjusted_score > best_score:
                    best_concept = concept
                    best_score = adjusted_score
        
        return best_concept, best_score