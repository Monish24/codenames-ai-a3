from players.codemaster import Codemaster
import random
import numpy as np
from model_manager import get_glove_model
from collections import defaultdict
import itertools

class CodemasterTreeOfThoughts(Codemaster):

    
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
        
        self.max_depth = 4  
        self.branch_factor = 7  
        self.pruning_threshold = 0.25  
        
        # Memory for learning
        self.successful_clues = {}  # clue -> success_rate
        self.failed_clues = set()
        
        # Enhanced vocabulary with difficulty levels
        vocab_items = sorted(self.model.key_to_index.items(), key=lambda kv: kv[1])
        self.vocabulary = self._create_tiered_vocabulary(vocab_items)
        
        # Word frequency and quality metrics
        self.word_frequency = {}
        self.word_quality = {}
        for i, (word, _) in enumerate(vocab_items[:12000]):
            if word.isalpha() and len(word) >= 3:
                self.word_frequency[word] = 1.0 - (i / 12000)
                self.word_quality[word] = self._assess_word_quality(word)
    
    def _create_tiered_vocabulary(self, vocab_items):
        """Create vocabulary tiers by frequency and quality"""
        vocab = {
            'common': [],    # Top 2000 - everyday words
            'moderate': [],  # 2000-6000 - educated vocabulary  
            'advanced': []   # 6000-12000 - sophisticated words
        }
        
        for i, (word, _) in enumerate(vocab_items[:12000]):
            if self._is_valid_vocabulary_word(word):
                if i < 2000:
                    vocab['common'].append(word)
                elif i < 6000:
                    vocab['moderate'].append(word)
                else:
                    vocab['advanced'].append(word)
        
        return vocab
    
    def _is_valid_vocabulary_word(self, word):
        """Enhanced word validation"""
        if not word.isalpha() or len(word) < 3 or len(word) > 15:
            return False
        
        # Filter out likely proper nouns, abbreviations, etc.
        if word[0].isupper() and len(word) > 4:
            return False
        
        # Filter words with unusual character patterns
        if len(set(word)) < len(word) * 0.5:  # Too many repeated chars
            return False
        
        # Filter likely foreign words or technical terms
        unusual_patterns = ['ph', 'gh', 'sch', 'chr', 'ps', 'gn']
        if any(pattern in word.lower() for pattern in unusual_patterns):
            return False
            
        return True
    
    def _assess_word_quality(self, word):
        """Assess how good a word is as a potential clue"""
        score = 0.5  # Base score
        
        # Length preference (4-8 chars optimal)
        if 4 <= len(word) <= 8:
            score += 0.2
        elif len(word) < 4 or len(word) > 10:
            score -= 0.2
        
        # Common ending patterns that suggest good concepts
        good_endings = ['ing', 'tion', 'ness', 'ment', 'able', 'ful']
        if any(word.endswith(ending) for ending in good_endings):
            score += 0.15
        
        # Vowel/consonant balance
        vowels = sum(1 for c in word.lower() if c in 'aeiou')
        if 0.2 <= vowels / len(word) <= 0.5:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def set_game_state(self, words_on_board, key_grid):
        """Set the current game state"""
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
    
    def _safe_word_similarity(self, w1, w2, default=0.0):
        """Safe similarity calculation with enhanced error handling"""
        try:
            w1, w2 = w1.lower().strip(), w2.lower().strip()
            
            if not w1 or not w2 or w1 == w2:
                return default
                
            if w1 not in self.model.key_to_index or w2 not in self.model.key_to_index:
                # Fallback to simple string similarity
                return self._string_similarity(w1, w2) * 0.2
                
            v1 = self.model[w1]
            v2 = self.model[w2]
            
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return default
                
            similarity = float(np.dot(v1, v2) / (norm1 * norm2))
            return max(-1.0, min(1.0, similarity))  # Clamp to valid range
            
        except Exception as e:
            print(f"Similarity calculation failed for '{w1}', '{w2}': {e}")
            return default
    
    def _string_similarity(self, w1, w2):
        """Simple string similarity as fallback"""
        if not w1 or not w2:
            return 0.0
        
        # Jaccard similarity of character bigrams
        bigrams1 = set(w1[i:i+2] for i in range(len(w1)-1))
        bigrams2 = set(w2[i:i+2] for i in range(len(w2)-1))
        
        if not bigrams1 and not bigrams2:
            return 1.0 if w1 == w2 else 0.0
        
        intersection = bigrams1 & bigrams2
        union = bigrams1 | bigrams2
        
        return len(intersection) / len(union) if union else 0.0
    
    class ThoughtNode:
        """Enhanced thought node with better tracking"""
        def __init__(self, reasoning_path, target_words, clue_candidates=None, 
                     score=0.0, depth=0, reasoning_type="unknown"):
            self.reasoning_path = reasoning_path
            self.target_words = target_words
            self.clue_candidates = clue_candidates or []
            self.score = score
            self.depth = depth
            self.reasoning_type = reasoning_type
            self.children = []
            self.is_pruned = False
            self.exploration_count = 0
            self.success_history = []
        
        def add_child(self, child_node):
            """Add a child thought node"""
            child_node.depth = self.depth + 1
            self.children.append(child_node)
        
        def prune(self, reason="low_score"):
            """Mark this branch as pruned with reason"""
            self.is_pruned = True
            self.prune_reason = reason
        
        def get_best_candidates(self, n=3):
            """Get top N candidates from this node"""
            if not self.clue_candidates:
                return []
            return sorted(self.clue_candidates, key=lambda x: x['score'], reverse=True)[:n]
    
    def _generate_reasoning_paths(self, target_words):
        """Generate enhanced reasoning paths"""
        reasoning_paths = []
        
        # Path 1: Direct semantic similarity (Level 1 - Basic)
        reasoning_paths.append({
            'name': 'direct_similarity',
            'description': 'Find words directly similar to targets',
            'method': self._explore_direct_similarity,
            'priority': 1
        })
        
        # Path 2: Categorical reasoning (Level 2 - Intermediate)
        reasoning_paths.append({
            'name': 'categorical_reasoning',
            'description': 'Find category or type that encompasses targets',
            'method': self._explore_categorical_reasoning,
            'priority': 2
        })
        
        # Path 3: Functional/usage reasoning (Level 2 - Intermediate)
        reasoning_paths.append({
            'name': 'functional_reasoning',
            'description': 'Find common function or usage patterns',
            'method': self._explore_functional_reasoning,
            'priority': 2
        })
        
        # Path 4: Contextual/environmental reasoning (Level 3 - Advanced)
        reasoning_paths.append({
            'name': 'contextual_reasoning',
            'description': 'Find shared contexts or environments',
            'method': self._explore_contextual_reasoning,
            'priority': 3
        })
        
        # Path 5: Temporal/sequential reasoning (Level 3 - Advanced)
        reasoning_paths.append({
            'name': 'temporal_reasoning',
            'description': 'Find temporal or sequential connections',
            'method': self._explore_temporal_reasoning,
            'priority': 3
        })
        
        # Path 6: Metaphorical/abstract reasoning (Level 4 - Expert)
        reasoning_paths.append({
            'name': 'metaphorical_reasoning',
            'description': 'Find metaphorical or abstract connections',
            'method': self._explore_metaphorical_reasoning,
            'priority': 4
        })
        
        # Path 7: Elimination/contrast reasoning (Level 4 - Expert)
        reasoning_paths.append({
            'name': 'elimination_reasoning',
            'description': 'Find words that connect targets while avoiding dangers',
            'method': self._explore_elimination_reasoning,
            'priority': 4
        })
        
        return reasoning_paths
    
    def _explore_direct_similarity(self, target_words, avoid_words):
        """Level 1: Direct similarity exploration with batching"""
        candidates = []
        
        # Batch process similar words for efficiency
        all_neighbors = defaultdict(lambda: {'targets': [], 'similarities': [], 'scores': []})
        
        for target in target_words:
            if target not in self.model.key_to_index:
                continue
                
            try:
                similar_words = self.model.most_similar(positive=[target], topn=25)
                for word, sim in similar_words:
                    if self._is_candidate_valid(word, avoid_words):
                        all_neighbors[word]['targets'].append(target)
                        all_neighbors[word]['similarities'].append(sim)
                        all_neighbors[word]['scores'].append(sim * self.word_quality.get(word, 0.5))
            except Exception as e:
                print(f"Error finding similar words for {target}: {e}")
                continue
        
        # Process candidates
        for word, data in all_neighbors.items():
            if len(data['targets']) >= 1:  # At least connects to 1 target
                avg_sim = np.mean(data['similarities'])
                max_sim = max(data['similarities'])
                quality_bonus = self.word_quality.get(word, 0.5) * 0.2
                frequency_bonus = self.word_frequency.get(word, 0.5) * 0.1
                
                # Bonus for connecting multiple targets
                multi_bonus = 0.8 if len(data['targets']) > 1 else 0
                
                final_score = avg_sim + quality_bonus + frequency_bonus + multi_bonus
                
                candidates.append({
                    'clue': word,
                    'targets': data['targets'],
                    'score': final_score,
                    'connections': len(data['targets']),
                    'reasoning': f'Direct similarity to {len(data["targets"])} targets',
                    'similarity_details': dict(zip(data['targets'], data['similarities']))
                })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _explore_categorical_reasoning(self, target_words, avoid_words):
        """Level 2: Find categorical connections"""
        candidates = []
        
        # Common category indicators
        category_words = [
            'animal', 'food', 'plant', 'tool', 'vehicle', 'building', 'color',
            'sport', 'game', 'music', 'art', 'science', 'nature', 'body',
            'clothing', 'furniture', 'weapon', 'instrument', 'container'
        ]
        
        for category in category_words:
            if category in self.model.key_to_index and category not in self.words_on_board:
                category_score = 0
                connected_targets = []
                
                for target in target_words:
                    sim = self._safe_word_similarity(category, target)
                    if sim > 0.25:  # Reasonable connection threshold
                        category_score += sim
                        connected_targets.append(target)
                
                if len(connected_targets) >= 2 or (len(target_words) == 1 and connected_targets):
                    avg_score = category_score / len(target_words)
                    quality_bonus = self.word_quality.get(category, 0.5) * 0.15
                    
                    candidates.append({
                        'clue': category,
                        'targets': connected_targets,
                        'score': avg_score + quality_bonus,
                        'connections': len(connected_targets),
                        'reasoning': f'Category connection to {len(connected_targets)} targets'
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _explore_functional_reasoning(self, target_words, avoid_words):
        """Level 2: Find functional/usage connections"""
        candidates = []
        
        # Function/usage indicators
        function_words = [
            'use', 'tool', 'work', 'make', 'create', 'build', 'help',
            'move', 'carry', 'hold', 'protect', 'clean', 'cook', 'play',
            'communicate', 'transport', 'store', 'display', 'measure'
        ]
        
        for func_word in function_words:
            if (func_word in self.model.key_to_index and 
                func_word not in self.words_on_board and
                func_word.lower() not in self.used_clues):
                
                func_score = 0
                connected_targets = []
                
                for target in target_words:
                    sim = self._safe_word_similarity(func_word, target)
                    if sim > 0.2:
                        func_score += sim
                        connected_targets.append(target)
                
                if connected_targets:
                    avg_score = func_score / len(target_words)
                    candidates.append({
                        'clue': func_word,
                        'targets': connected_targets,
                        'score': avg_score,
                        'connections': len(connected_targets),
                        'reasoning': f'Functional connection to {len(connected_targets)} targets'
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _explore_contextual_reasoning(self, target_words, avoid_words):
        """Level 3: Find shared contexts or environments"""
        candidates = []
        
        # Context/environment words
        context_words = [
            'home', 'kitchen', 'office', 'school', 'hospital', 'farm',
            'forest', 'ocean', 'city', 'country', 'space', 'underground',
            'indoor', 'outdoor', 'public', 'private', 'modern', 'ancient'
        ]
        
        for context in context_words:
            if (context in self.model.key_to_index and 
                context not in self.words_on_board and
                context.lower() not in self.used_clues):
                
                context_score = 0
                connected_targets = []
                
                for target in target_words:
                    sim = self._safe_word_similarity(context, target)
                    if sim > 0.15:  # Lower threshold for contextual connections
                        context_score += sim
                        connected_targets.append(target)
                
                if len(connected_targets) >= 2:
                    avg_score = context_score / len(target_words)
                    candidates.append({
                        'clue': context,
                        'targets': connected_targets,
                        'score': avg_score,
                        'connections': len(connected_targets),
                        'reasoning': f'Contextual connection to {len(connected_targets)} targets'
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _explore_temporal_reasoning(self, target_words, avoid_words):
        """Level 3: Find temporal/sequential connections"""
        candidates = []
        
        # Temporal indicators
        temporal_words = [
            'before', 'after', 'during', 'first', 'last', 'early', 'late',
            'past', 'future', 'old', 'new', 'process', 'sequence', 'order',
            'time', 'period', 'era', 'moment', 'cycle', 'pattern'
        ]
        
        for temporal in temporal_words:
            if (temporal in self.model.key_to_index and 
                temporal not in self.words_on_board and
                temporal.lower() not in self.used_clues):
                
                temporal_score = 0
                connected_targets = []
                
                for target in target_words:
                    sim = self._safe_word_similarity(temporal, target)
                    if sim > 0.2:
                        temporal_score += sim
                        connected_targets.append(target)
                
                if connected_targets:
                    avg_score = temporal_score / len(target_words)
                    candidates.append({
                        'clue': temporal,
                        'targets': connected_targets,
                        'score': avg_score,
                        'connections': len(connected_targets),
                        'reasoning': f'Temporal connection to {len(connected_targets)} targets'
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _explore_metaphorical_reasoning(self, target_words, avoid_words):
        """Level 4: Find metaphorical/abstract connections"""
        candidates = []
        
        if len(target_words) < 2:
            return candidates
        
        # Try to find abstract concepts that metaphorically connect targets
        abstract_concepts = [
            'strength', 'power', 'beauty', 'speed', 'size', 'weight',
            'bright', 'dark', 'hard', 'soft', 'hot', 'cold', 'rough', 'smooth',
            'freedom', 'peace', 'energy', 'life', 'growth', 'change'
        ]
        
        for concept in abstract_concepts:
            if (concept in self.model.key_to_index and 
                concept not in self.words_on_board and
                concept.lower() not in self.used_clues):
                
                concept_score = 0
                connected_targets = []
                
                for target in target_words:
                    sim = self._safe_word_similarity(concept, target)
                    if sim > 0.15:  # Lower threshold for abstract connections
                        concept_score += sim
                        connected_targets.append(target)
                
                # Require connection to multiple targets for abstract concepts
                if len(connected_targets) >= 2:
                    avg_score = concept_score / len(target_words)
                    # Bonus for abstract thinking
                    abstract_bonus = 0.1
                    
                    candidates.append({
                        'clue': concept,
                        'targets': connected_targets,
                        'score': avg_score + abstract_bonus,
                        'connections': len(connected_targets),
                        'reasoning': f'Abstract/metaphorical connection to {len(connected_targets)} targets'
                    })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _explore_elimination_reasoning(self, target_words, avoid_words):
        """Level 4: Enhanced elimination reasoning with safety focus"""
        candidates = []
        
        # Get potential candidates from direct similarity first
        potential_candidates = set()
        for target in target_words:
            if target not in self.model.key_to_index:
                continue
            try:
                similar_words = self.model.most_similar(positive=[target], topn=30)
                for word, _ in similar_words:
                    if self._is_candidate_valid(word, avoid_words):
                        potential_candidates.add(word)
            except:
                continue
        
        # Evaluate each candidate for safety and effectiveness
        for candidate in potential_candidates:
            # Calculate target connections
            target_scores = []
            connected_targets = []
            
            for target in target_words:
                sim = self._safe_word_similarity(candidate, target)
                target_scores.append(sim)
                if sim > 0.3:
                    connected_targets.append(target)
            
            if not connected_targets:
                continue
            
            # Calculate danger scores
            danger_score = 0
            max_danger = 0
            
            for avoid_word in avoid_words:
                if avoid_word:
                    danger_sim = self._safe_word_similarity(candidate, avoid_word)
                    danger_score += danger_sim
                    max_danger = max(max_danger, danger_sim)
            
            # Special assassin check
            assassin_sim = 0
            if self.assassin_word:
                assassin_sim = self._safe_word_similarity(candidate, self.assassin_word)
                danger_score += assassin_sim * 2  # Double weight for assassin
                max_danger = max(max_danger, assassin_sim)
            
            # Skip very dangerous candidates
            if assassin_sim > 0.35 or max_danger > 0.4:
                continue
            
            # Calculate safety-adjusted score
            avg_target_score = np.mean(target_scores)
            safety_multiplier = max(0.1, 1.0 - danger_score)
            
            final_score = avg_target_score * safety_multiplier
            
            # Bonus for multiple connections
            if len(connected_targets) > 1:
                final_score += 0.8
            
            candidates.append({
                'clue': candidate,
                'targets': connected_targets,
                'score': final_score,
                'connections': len(connected_targets),
                'reasoning': f'Safe connection to {len(connected_targets)} targets (safety: {safety_multiplier:.2f})',
                'safety_score': safety_multiplier
            })
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:self.branch_factor]
    
    def _is_candidate_valid(self, word, avoid_words):
        """Enhanced candidate validation"""
        if not word or not word.isalpha() or len(word) < 3:
            return False
        
        if word in self.words_on_board or word.lower() in self.used_clues:
            return False
        
        # Check if word is substring of or contains board words
        for board_word in self.words_on_board:
            if word.lower() in board_word.lower() or board_word.lower() in word.lower():
                return False
        
        # Check if it's in our failed clues
        if word.lower() in self.failed_clues:
            return False
        
        return True
    
    def _build_thought_tree(self, target_words, avoid_words):
        """Build and explore the enhanced tree of thoughts"""
        reasoning_paths = self._generate_reasoning_paths(target_words)
        all_candidates = []
        
        # Explore each reasoning path with priority ordering
        reasoning_paths.sort(key=lambda x: x['priority'])
        
        for path_info in reasoning_paths:
            try:
                path_method = path_info['method']
                path_name = path_info['name']
                
                candidates = path_method(target_words, avoid_words)
                
                # Create thought nodes and add to candidates
                for candidate in candidates:
                    if candidate['score'] > self.pruning_threshold:
                        # Adjust score based on memory
                        adjusted_score = self._adjust_score_with_memory(
                            candidate['clue'], candidate['score']
                        )
                        
                        candidate['score'] = adjusted_score
                        candidate['reasoning_path'] = path_name
                        candidate['priority'] = path_info['priority']
                        
                        all_candidates.append(candidate)
            
            except Exception as e:
                print(f"Error in reasoning path {path_info['name']}: {e}")
                continue
        
        # Sort by adjusted score and return top candidates
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        return all_candidates
    
    def _adjust_score_with_memory(self, clue, base_score):
        """Adjust score based on historical performance"""
        if clue.lower() in self.successful_clues:
            # Boost previously successful clues
            success_rate = self.successful_clues[clue.lower()]
            return base_score * (1.0 + success_rate * 0.3)
        elif clue.lower() in self.failed_clues:
            # Penalize previously failed clues
            return base_score * 0.7
        
        return base_score
    
    def remember_clue_outcome(self, clue, success_rate):
        """Learn from clue outcomes for future reference"""
        clue_lower = clue.lower()
        if success_rate > 0.6:
            self.successful_clues[clue_lower] = success_rate
        elif success_rate < 0.3:
            self.failed_clues.add(clue_lower)
    
    def get_clue(self):
        """Generate clue using enhanced Tree of Thoughts reasoning"""
        if not self.my_words:
            return "PASS", 0
        
        remaining_words = [w for w in self.my_words if not w.startswith("*")]
        if not remaining_words:
            return "PASS", 0
        
        avoid_words = (self.opponent_words + self.civilian_words + 
                      ([self.assassin_word] if self.assassin_word else []))
        
        best_candidate = None
        best_score = -1
        
        # Try different target combinations with curriculum learning approach
        combinations_to_try = []
        
        # Single words (easiest)
        for word in remaining_words:
            combinations_to_try.append(([word], 1))
        
        # Pairs (moderate difficulty)
        if len(remaining_words) >= 2:
            for i in range(len(remaining_words)):
                for j in range(i + 1, min(i + 4, len(remaining_words))):
                    pair = [remaining_words[i], remaining_words[j]]
                    combinations_to_try.append((pair, 2))
        
        # Triples (hardest)
        if len(remaining_words) >= 3:
            for i in range(min(3, len(remaining_words) - 2)):
                triple = remaining_words[i:i+3]
                combinations_to_try.append((triple, 3))
        
        # Explore tree of thoughts for each combination
        for target_combination, target_count in combinations_to_try:
            try:
                candidates = self._build_thought_tree(target_combination, avoid_words)
                
                for candidate in candidates[:5]:  # Top 5 from each tree
                    # Enhanced scoring with curriculum bonus
                    curriculum_bonus = 0.1 * (target_count - 1)  # Bonus for harder problems
                    multi_bonus = 0.8 if candidate['connections'] > 1 else 0
                    
                    adjusted_score = candidate['score'] + curriculum_bonus + multi_bonus
                    
                    if adjusted_score > best_score:
                        best_candidate = candidate
                        best_score = adjusted_score
                        
            except Exception as e:
                print(f"Error processing combination {target_combination}: {e}")
                continue
        
        # Return best result
        if best_candidate and best_score > 0.2:  # Minimum quality threshold
            clue = best_candidate['clue'].upper()
            num = len(best_candidate['targets'])
            
            self.used_clues.add(clue.lower())
            
            print(f"ToT Reasoning: {best_candidate.get('reasoning', 'Unknown')}")
            print(f"ToT Path: {best_candidate.get('reasoning_path', 'Unknown')}")
            
            return clue, num
        
        # Enhanced fallback
        if remaining_words:
            # Try a simple safe word
            safe_fallbacks = ["CONCEPT", "IDEA", "THING", "ITEM", "ELEMENT"]
            for fallback in safe_fallbacks:
                if not self._is_candidate_valid(fallback, avoid_words):
                    continue
                
                # Check if it's reasonably safe
                danger_score = sum(self._safe_word_similarity(fallback, avoid) 
                                 for avoid in avoid_words if avoid)
                if danger_score < 0.5:
                    return fallback, 1
        
        return "THOUGHT", 1