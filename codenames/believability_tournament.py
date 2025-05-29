import json
import numpy as np
import time
import re
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
import random
import math

from tournament import EnhancedTournamentManager, ClueRecord, GuessRecord, BelievabilityAnalyzer
from model_manager import get_glove_model

@dataclass
class EnhancedClueMetrics:
    """Enhanced clue metrics with all believability components"""
    clue_word: str
    codemaster_name: str
    target_words: List[str]
    guessed_words: List[str]
    game_seed: int
    
    # Basic performance metrics
    clue_efficiency: float  # target_cards / revealed_cards
    success_rate: float  # How many intended targets were guessed
    safety_score: float  # Avoided dangerous cards
    
    # Believability metrics
    frequency_score: float  # Word commonness/typicality
    semantic_coherence: float  # Target similarity - distractor similarity
    human_likeness: float  # How human-like the clue appears
    diversity_contribution: float  # How this clue affects agent's diversity
    safety_toxicity: float  # Toxicity/safety assessment
    
    # Derived scores
    composite_believability: float
    z_scores: Dict[str, float]  # Z-scores against human baseline

@dataclass
class AgentBelievabilityProfile:
    """Complete believability profile for an agent"""
    agent_name: str
    agent_type: str
    
    # Aggregated metrics
    avg_frequency_score: float = 0.0
    avg_semantic_coherence: float = 0.0
    avg_human_likeness: float = 0.0
    diversity_score: float = 0.0  # Type/token ratio, self-BLEU
    avg_safety_score: float = 0.0
    
    # Composite and normalized scores
    composite_believability: float = 0.0
    z_normalized_scores: Dict[str, float] = None
    
    # Performance correlation
    performance_believability_correlation: float = 0.0
    
    # Detailed analysis
    clue_metrics: List[EnhancedClueMetrics] = None
    
    def __post_init__(self):
        if self.z_normalized_scores is None:
            self.z_normalized_scores = {}
        if self.clue_metrics is None:
            self.clue_metrics = []

class AdvancedBelievabilityAnalyzer(BelievabilityAnalyzer):
    """Advanced believability analyzer with enhanced metrics"""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced human baseline statistics (would be loaded from corpus)
        self.human_baseline_stats = {
            'frequency': {'mean': 0.006, 'std': 0.004, 'samples': []},
            'coherence': {'mean': 0.62, 'std': 0.18, 'samples': []},
            'human_likeness': {'mean': 0.73, 'std': 0.12, 'samples': []},
            'diversity': {'mean': 0.85, 'std': 0.08, 'samples': []},
            'safety': {'mean': 0.96, 'std': 0.04, 'samples': []}
        }
        
        # Initialize additional models if available
        self._init_advanced_models()
    
    def _init_advanced_models(self):
        """Initialize additional models for advanced analysis"""
        try:
            # Try to load additional models for more sophisticated analysis
            self.glove_model = get_glove_model()
            print("Loaded GloVe model for advanced believability analysis")
        except Exception as e:
            print(f"Could not load GloVe model: {e}")
            self.glove_model = None
        
        # Placeholder for other models (GPT-2 for perplexity, toxicity classifiers, etc.)
        self.toxicity_model = None
        self.perplexity_model = None
    
    def calculate_enhanced_frequency_score(self, clue: str) -> Tuple[float, Dict[str, Any]]:
        """Enhanced frequency calculation with detailed breakdown"""
        clue_lower = clue.lower()
        
        # Base frequency from multiple sources
        base_freq = self.word_frequencies.get(clue_lower, 0.0001)
        
        # TF-IDF style rarity assessment
        # In real implementation, this would use actual corpus statistics
        doc_frequency = 0.1  # Placeholder - would be actual document frequency
        idf_score = math.log(1.0 / (doc_frequency + 1e-10))
        
        # Surprisal calculation: -log P(word)
        surprisal = -math.log(base_freq + 1e-10)
        
        # Length-based adjustments
        length_factor = 1.0
        if len(clue) <= 3:
            length_factor = 0.7  # Very short words less typical
        elif len(clue) > 12:
            length_factor = 0.6  # Very long words less typical
        elif 4 <= len(clue) <= 8:
            length_factor = 1.2  # Optimal length bonus
        
        # Pattern-based adjustments
        pattern_factor = 1.0
        unusual_patterns = ['ph', 'gh', 'sch', 'chr', 'ps', 'gn', 'wr']
        if any(pattern in clue_lower for pattern in unusual_patterns):
            pattern_factor = 0.8
        
        # Common endings that suggest good clue words
        good_endings = ['ing', 'tion', 'ness', 'ment', 'able', 'ful', 'ly']
        if any(clue_lower.endswith(ending) for ending in good_endings):
            pattern_factor = min(1.5, pattern_factor * 1.3)
        
        # Final frequency score (higher = more typical/frequent)
        raw_score = base_freq * length_factor * pattern_factor
        normalized_score = min(1.0, raw_score * 100)  # Scale to 0-1
        
        details = {
            'base_frequency': base_freq,
            'surprisal': surprisal,
            'idf_score': idf_score,
            'length_factor': length_factor,
            'pattern_factor': pattern_factor,
            'raw_score': raw_score
        }
        
        return normalized_score, details
    
    def calculate_enhanced_semantic_coherence(self, clue: str, targets: List[str], 
                                            all_board_words: List[str]) -> Tuple[float, Dict[str, Any]]:
        """Enhanced semantic coherence with detailed analysis"""
        if not self.sentence_model or not targets:
            return 0.5, {'error': 'No model or targets available'}
        
        try:
            # Separate distractors by type
            distractors = [w for w in all_board_words if w not in targets]
            
            # Encode all words
            all_words = [clue] + targets + distractors
            embeddings = self.sentence_model.encode(all_words)
            
            clue_embedding = embeddings[0]
            target_embeddings = embeddings[1:len(targets)+1]
            distractor_embeddings = embeddings[len(targets)+1:] if distractors else np.array([])
            
            # Calculate target similarities
            target_sims = []
            for t_emb in target_embeddings:
                sim = np.dot(clue_embedding, t_emb) / (
                    np.linalg.norm(clue_embedding) * np.linalg.norm(t_emb)
                )
                target_sims.append(max(0, sim))  # Only positive similarities
            
            # Calculate distractor similarities
            distractor_sims = []
            if len(distractor_embeddings) > 0:
                for d_emb in distractor_embeddings:
                    sim = np.dot(clue_embedding, d_emb) / (
                        np.linalg.norm(clue_embedding) * np.linalg.norm(d_emb)
                    )
                    distractor_sims.append(max(0, sim))
            
            avg_target_sim = np.mean(target_sims) if target_sims else 0
            avg_distractor_sim = np.mean(distractor_sims) if distractor_sims else 0
            max_target_sim = np.max(target_sims) if target_sims else 0
            max_distractor_sim = np.max(distractor_sims) if distractor_sims else 0
            
            # Enhanced coherence calculation
            # Primary metric: target_sim - distractor_sim
            coherence_diff = avg_target_sim - avg_distractor_sim
            
            # Bonus for strong target connections
            target_strength = avg_target_sim
            
            # Penalty for dangerous distractor similarities
            danger_penalty = max(0, max_distractor_sim - 0.3) * 2
            
            # Final coherence score
            coherence = (coherence_diff + target_strength - danger_penalty)
            coherence = max(0, min(1, (coherence + 1) / 2))  # Normalize to 0-1
            
            details = {
                'avg_target_similarity': avg_target_sim,
                'max_target_similarity': max_target_sim,
                'avg_distractor_similarity': avg_distractor_sim,
                'max_distractor_similarity': max_distractor_sim,
                'coherence_difference': coherence_diff,
                'target_strength': target_strength,
                'danger_penalty': danger_penalty,
                'num_targets': len(targets),
                'num_distractors': len(distractors)
            }
            
            return coherence, details
            
        except Exception as e:
            return 0.5, {'error': str(e)}
    
    def calculate_enhanced_human_likeness(self, clue: str) -> Tuple[float, Dict[str, Any]]:
        """Enhanced human-likeness with multiple factors"""
        
        score = 0.5  # Base score
        factors = {}
        
        # Length preferences (based on human gameplay analysis)
        length_score = 0.5
        if 4 <= len(clue) <= 8:
            length_score = 0.9
        elif 3 <= len(clue) <= 10:
            length_score = 0.7
        elif len(clue) < 3 or len(clue) > 12:
            length_score = 0.2
        factors['length_score'] = length_score
        
        # Morphological patterns humans prefer
        morphology_score = 0.5
        good_patterns = ['ing', 'tion', 'ness', 'ment', 'able', 'ful', 'ly', 'er', 'est']
        if any(clue.lower().endswith(pattern) for pattern in good_patterns):
            morphology_score = 0.8
        
        # Avoid overly technical patterns
        technical_patterns = ['ology', 'ography', 'ification', 'ization']
        if any(pattern in clue.lower() for pattern in technical_patterns):
            morphology_score = 0.3
        factors['morphology_score'] = morphology_score
        
        # Letter pattern analysis
        vowel_ratio = sum(1 for c in clue.lower() if c in 'aeiou') / len(clue)
        vowel_score = 0.8 if 0.2 <= vowel_ratio <= 0.5 else 0.4
        factors['vowel_score'] = vowel_score
        
        # Consonant cluster analysis (humans avoid complex clusters)
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', clue.lower())
        cluster_penalty = len(consonant_clusters) * 0.1
        factors['consonant_cluster_penalty'] = cluster_penalty
        
        # Common vs rare letter combinations
        common_bigrams = ['th', 'he', 'in', 'er', 'an', 're', 'ed', 'nd', 'on', 'en']
        bigrams_in_word = [clue.lower()[i:i+2] for i in range(len(clue)-1)]
        common_bigram_count = sum(1 for bg in bigrams_in_word if bg in common_bigrams)
        bigram_score = min(1.0, common_bigram_count / len(bigrams_in_word)) if bigrams_in_word else 0.5
        factors['bigram_score'] = bigram_score
        
        # Frequency-based human preference (common words more human-like)
        word_freq = self.word_frequencies.get(clue.lower(), 0.0001)
        freq_score = min(1.0, word_freq * 1000)  # Scale appropriately
        factors['frequency_human_score'] = freq_score
        
        # Composite human-likeness score
        composite = (
            length_score * 0.25 +
            morphology_score * 0.25 +
            vowel_score * 0.15 +
            bigram_score * 0.15 +
            freq_score * 0.20 -
            cluster_penalty
        )
        
        final_score = max(0, min(1, composite))
        factors['composite_score'] = final_score
        
        return final_score, factors
    
    def calculate_diversity_metrics(self, agent_clues: List[str]) -> Tuple[float, Dict[str, Any]]:
        """Calculate diversity using multiple metrics"""
        if not agent_clues:
            return 1.0, {'error': 'No clues provided'}
        
        clues_lower = [clue.lower() for clue in agent_clues]
        
        # Type/token ratio
        unique_clues = len(set(clues_lower))
        total_clues = len(clues_lower)
        type_token_ratio = unique_clues / total_clues
        
        # Self-BLEU approximation (simplified)
        # Count n-gram overlaps between clues
        def get_ngrams(text, n):
            return [text[i:i+n] for i in range(len(text)-n+1)]
        
        # Calculate average self-similarity
        self_similarities = []
        for i, clue1 in enumerate(clues_lower):
            similarities_with_others = []
            for j, clue2 in enumerate(clues_lower):
                if i != j:
                    # Character-level 3-grams
                    ngrams1 = set(get_ngrams(clue1, 3))
                    ngrams2 = set(get_ngrams(clue2, 3))
                    
                    if ngrams1 and ngrams2:
                        jaccard_sim = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
                        similarities_with_others.append(jaccard_sim)
            
            if similarities_with_others:
                self_similarities.append(np.mean(similarities_with_others))
        
        avg_self_similarity = np.mean(self_similarities) if self_similarities else 0
        diversity_from_self_bleu = 1 - avg_self_similarity
        
        # Repetition penalty
        clue_counts = Counter(clues_lower)
        max_repetition = max(clue_counts.values())
        repetition_diversity = 1 - (max_repetition - 1) / total_clues
        
        # Length diversity
        lengths = [len(clue) for clue in agent_clues]
        length_std = np.std(lengths) if len(lengths) > 1 else 0
        length_diversity = min(1.0, length_std / 3)  # Normalize
        
        # Final diversity score
        diversity_score = (
            type_token_ratio * 0.4 +
            diversity_from_self_bleu * 0.3 +
            repetition_diversity * 0.2 +
            length_diversity * 0.1
        )
        
        details = {
            'type_token_ratio': type_token_ratio,
            'unique_clues': unique_clues,
            'total_clues': total_clues,
            'avg_self_similarity': avg_self_similarity,
            'diversity_from_self_bleu': diversity_from_self_bleu,
            'repetition_diversity': repetition_diversity,
            'length_diversity': length_diversity,
            'most_common_clue': clue_counts.most_common(1)[0] if clue_counts else None
        }
        
        return diversity_score, details
    
    def calculate_enhanced_safety_score(self, clue: str) -> Tuple[float, Dict[str, Any]]:
        """Enhanced safety/toxicity assessment"""
        
        safety_score = 1.0  # Start with perfect safety
        factors = {}
        
        # Basic offensive word detection
        offensive_words = [
            'hate', 'kill', 'die', 'dead', 'blood', 'murder', 'bomb', 'gun', 'war',
            'stupid', 'idiot', 'moron', 'damn', 'hell', 'shit', 'fuck', 'bitch'
        ]
        
        clue_lower = clue.lower()
        offensive_matches = [word for word in offensive_words if word in clue_lower]
        offensive_penalty = len(offensive_matches) * 0.3
        factors['offensive_matches'] = offensive_matches
        factors['offensive_penalty'] = offensive_penalty
        
        # Violence/aggression indicators
        violence_words = ['attack', 'fight', 'hurt', 'pain', 'wound', 'strike', 'hit']
        violence_matches = [word for word in violence_words if word in clue_lower]
        violence_penalty = len(violence_matches) * 0.2
        factors['violence_matches'] = violence_matches
        factors['violence_penalty'] = violence_penalty
        
        # Discrimination/bias indicators
        bias_words = ['race', 'gender', 'religion', 'ethnic', 'nationality']
        # This is very simplified - real implementation would be more sophisticated
        bias_penalty = 0.1 if any(word in clue_lower for word in bias_words) else 0
        factors['bias_penalty'] = bias_penalty
        
        # All caps penalty (shouting)
        caps_penalty = 0.1 if clue.isupper() and len(clue) > 3 else 0
        factors['caps_penalty'] = caps_penalty
        
        # Calculate final safety score
        total_penalty = offensive_penalty + violence_penalty + bias_penalty + caps_penalty
        final_safety = max(0, min(1, safety_score - total_penalty))
        
        factors['total_penalty'] = total_penalty
        factors['final_safety_score'] = final_safety
        
        return final_safety, factors
    
    def calculate_z_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate z-scores against human baseline"""
        z_scores = {}
        
        for metric_name, value in metrics.items():
            if metric_name in self.human_baseline_stats:
                baseline = self.human_baseline_stats[metric_name]
                z_score = (value - baseline['mean']) / baseline['std']
                z_scores[f'{metric_name}_zscore'] = z_score
        
        return z_scores
    
    def calculate_composite_believability_enhanced(
        self, 
        clue_record: ClueRecord,
        agent_all_clues: List[str],
        board_words: List[str]
    ) -> EnhancedClueMetrics:
        """Calculate comprehensive believability metrics for a clue"""
        
        clue = clue_record.clue_word
        targets = clue_record.target_words
        
        # Calculate each component with detailed breakdown
        frequency_score, freq_details = self.calculate_enhanced_frequency_score(clue)
        coherence_score, coherence_details = self.calculate_enhanced_semantic_coherence(
            clue, targets, board_words
        )
        human_score, human_details = self.calculate_enhanced_human_likeness(clue)
        diversity_score, diversity_details = self.calculate_diversity_metrics(agent_all_clues)
        safety_score, safety_details = self.calculate_enhanced_safety_score(clue)
        
        # Calculate performance metrics
        clue_efficiency = (clue_record.target_cards / 
                          max(1, clue_record.revealed_cards))
        success_rate = (len([w for w in clue_record.guessed_words if w in targets]) / 
                       max(1, len(targets)))
        
        # Core believability metrics
        core_metrics = {
            'frequency': frequency_score,
            'coherence': coherence_score,
            'human_likeness': human_score,
            'diversity': diversity_score,
            'safety': safety_score
        }
        
        # Calculate z-scores
        z_scores = self.calculate_z_scores(core_metrics)
        
        # Composite believability with configurable weights
        composite = (
            coherence_score * 0.35 +      # Semantic coherence most important
            frequency_score * 0.20 +      # Word typicality
            human_score * 0.20 +          # Human-like patterns
            diversity_score * 0.15 +      # Diversity contribution
            safety_score * 0.10           # Safety/toxicity
        )
        
        # Create comprehensive metrics object
        enhanced_metrics = EnhancedClueMetrics(
            clue_word=clue,
            codemaster_name=clue_record.codemaster_name,
            target_words=targets,
            guessed_words=clue_record.guessed_words,
            game_seed=clue_record.game_seed,
            
            # Performance metrics
            clue_efficiency=clue_efficiency,
            success_rate=success_rate,
            safety_score=safety_score,
            
            # Believability metrics
            frequency_score=frequency_score,
            semantic_coherence=coherence_score,
            human_likeness=human_score,
            diversity_contribution=diversity_score,
            safety_toxicity=safety_score,
            
            # Composite scores
            composite_believability=composite,
            z_scores=z_scores
        )
        
        return enhanced_metrics

class EnhancedBelievabilityTournament(EnhancedTournamentManager):
    """Enhanced tournament with comprehensive believability analysis"""
    
    def __init__(self, tournament_name="Enhanced_Believability_Tournament", 
                 progress_callback=None, **kwargs):
        super().__init__(tournament_name, **kwargs)
        
        self.progress_callback = progress_callback
        self.believability_analyzer = AdvancedBelievabilityAnalyzer()
        
        # Enhanced data collection
        self.enhanced_clue_metrics: List[EnhancedClueMetrics] = []
        self.agent_believability_profiles: Dict[str, AgentBelievabilityProfile] = {}
        
        print("Enhanced Believability Tournament initialized")
    
    def run_tournament_with_believability(self, shuffle_matchups: bool = True):
        """Run tournament with enhanced believability tracking"""
        print(f"Starting Enhanced Believability Tournament: {self.tournament_name}")
        
        # Run base tournament
        results = self.run_tournament(shuffle_matchups)
        
        # Perform enhanced believability analysis
        print("Performing enhanced believability analysis...")
        self.analyze_enhanced_believability()
        
        # Generate comprehensive believability profiles
        self.generate_agent_believability_profiles()
        
        # Calculate performance-believability correlations
        self.analyze_performance_believability_correlations()
        
        # Update results with believability data
        enhanced_results = self.get_enhanced_results_with_believability(results)
        
        # Save enhanced results
        self.save_enhanced_believability_results(enhanced_results)
        
        return enhanced_results
    
    def analyze_enhanced_believability(self):
        """Perform comprehensive believability analysis on all clues"""
        print("Analyzing clue believability with enhanced metrics...")
        
        # Group clues by agent for diversity calculation
        agent_clues = defaultdict(list)
        
        # First pass: collect all clues per agent
        for clue_record in self.all_clue_records:
            agent_clues[clue_record.codemaster_name].append(clue_record.clue_word)
        
        # Second pass: analyze each clue with full context
        for clue_record in self.all_clue_records:
            agent_name = clue_record.codemaster_name
            
            # Get all clues for this agent (for diversity calculation)
            agent_all_clues = agent_clues[agent_name]
            
            # Approximate board words (in real implementation, this would be exact)
            board_words = clue_record.target_words + ['dummy'] * 20  # Simplified
            
            # Calculate enhanced metrics
            enhanced_metrics = self.believability_analyzer.calculate_composite_believability_enhanced(
                clue_record, agent_all_clues, board_words
            )
            
            self.enhanced_clue_metrics.append(enhanced_metrics)
        
        print(f"Analyzed {len(self.enhanced_clue_metrics)} clues with enhanced metrics")
    
    def generate_agent_believability_profiles(self):
        """Generate comprehensive believability profiles for each agent"""
        print("Generating agent believability profiles...")
        
        # Group enhanced metrics by agent
        agent_metrics = defaultdict(list)
        for metrics in self.enhanced_clue_metrics:
            agent_metrics[metrics.codemaster_name].append(metrics)
        
        # Generate profile for each codemaster
        for agent_name, metrics_list in agent_metrics.items():
            if not metrics_list:
                continue
            
            # Calculate aggregated scores
            avg_frequency = np.mean([m.frequency_score for m in metrics_list])
            avg_coherence = np.mean([m.semantic_coherence for m in metrics_list])
            avg_human_likeness = np.mean([m.human_likeness for m in metrics_list])
            avg_safety = np.mean([m.safety_toxicity for m in metrics_list])
            
            # Diversity is calculated across all clues
            all_clues = [m.clue_word for m in metrics_list]
            diversity_score, _ = self.believability_analyzer.calculate_diversity_metrics(all_clues)
            
            # Composite believability
            composite = np.mean([m.composite_believability for m in metrics_list])
            
            # Z-scores against human baseline
            core_metrics = {
                'frequency': avg_frequency,
                'coherence': avg_coherence,
                'human_likeness': avg_human_likeness,
                'diversity': diversity_score,
                'safety': avg_safety
            }
            z_scores = self.believability_analyzer.calculate_z_scores(core_metrics)
            
            # Create profile
            profile = AgentBelievabilityProfile(
                agent_name=agent_name,
                agent_type='codemaster',
                avg_frequency_score=avg_frequency,
                avg_semantic_coherence=avg_coherence,
                avg_human_likeness=avg_human_likeness,
                diversity_score=diversity_score,
                avg_safety_score=avg_safety,
                composite_believability=composite,
                z_normalized_scores=z_scores,
                clue_metrics=metrics_list
            )
            
            self.agent_believability_profiles[agent_name] = profile
        
        print(f"Generated believability profiles for {len(self.agent_believability_profiles)} agents")
    
    def analyze_performance_believability_correlations(self):
        """Analyze correlation between performance and believability"""
        print("Analyzing performance-believability correlations...")
        
        for agent_name, profile in self.agent_believability_profiles.items():
            if agent_name in self.agent_metrics:
                # Get performance metrics
                agent_performance = self.agent_metrics[agent_name]
                win_rate = agent_performance.wins / max(1, agent_performance.games_played)
                trueskill_mu = agent_performance.role_based_rating.mu
                
                # Calculate correlation with believability
                # This is simplified - in practice, you'd use proper correlation analysis
                performance_score = (win_rate + (trueskill_mu - 25) / 25) / 2
                believability_score = profile.composite_believability
                
                # Simple correlation approximation
                correlation = np.corrcoef([performance_score], [believability_score])[0, 1]
                if not np.isnan(correlation):
                    profile.performance_believability_correlation = correlation
        
        print("Performance-believability correlation analysis completed")
    
    def get_enhanced_results_with_believability(self, base_results):
        """Enhance base results with believability data"""
        
        # Add believability profiles to agent metrics
        for agent_name, profile in self.agent_believability_profiles.items():
            if agent_name in self.agent_metrics:
                agent_metrics = self.agent_metrics[agent_name]
                agent_metrics.believability_scores = {
                    'composite': profile.composite_believability,
                    'frequency': profile.avg_frequency_score,
                    'coherence': profile.avg_semantic_coherence,
                    'human_likeness': profile.avg_human_likeness,
                    'diversity': profile.diversity_score,
                    'safety': profile.avg_safety_score,
                    'z_scores': profile.z_normalized_scores,
                    'performance_correlation': profile.performance_believability_correlation
                }
                agent_metrics.composite_believability = profile.composite_believability
        
        # Update base results
        base_results.believability_analysis.update({
            'enhanced_metrics_count': len(self.enhanced_clue_metrics),
            'agent_profiles': {name: asdict(profile) for name, profile in self.agent_believability_profiles.items()},
            'metric_distributions': self._calculate_metric_distributions(),
            'top_believable_clues': self._get_top_believable_clues(),
            'performance_believability_analysis': self._get_performance_believability_summary()
        })
        
        return base_results
    
    def _calculate_metric_distributions(self) -> Dict[str, Dict[str, float]]:
        """Calculate distribution statistics for each metric"""
        if not self.enhanced_clue_metrics:
            return {}
        
        metrics = {
            'frequency': [m.frequency_score for m in self.enhanced_clue_metrics],
            'coherence': [m.semantic_coherence for m in self.enhanced_clue_metrics],
            'human_likeness': [m.human_likeness for m in self.enhanced_clue_metrics],
            'safety': [m.safety_toxicity for m in self.enhanced_clue_metrics],
            'composite': [m.composite_believability for m in self.enhanced_clue_metrics]
        }
        
        distributions = {}
        for metric_name, values in metrics.items():
            distributions[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        return distributions
    
    def _get_top_believable_clues(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N most believable clues"""
        sorted_clues = sorted(
            self.enhanced_clue_metrics, 
            key=lambda x: x.composite_believability, 
            reverse=True
        )
        
        return [
            {
                'clue': clue.clue_word,
                'codemaster': clue.codemaster_name,
                'targets': clue.target_words,
                'composite_believability': clue.composite_believability,
                'frequency_score': clue.frequency_score,
                'coherence_score': clue.semantic_coherence,
                'human_likeness': clue.human_likeness,
                'safety_score': clue.safety_toxicity
            }
            for clue in sorted_clues[:n]
        ]
    
    def _get_performance_believability_summary(self) -> Dict[str, Any]:
        """Get summary of performance-believability relationships"""
        correlations = []
        for profile in self.agent_believability_profiles.values():
            if not np.isnan(profile.performance_believability_correlation):
                correlations.append(profile.performance_believability_correlation)
        
        if not correlations:
            return {'message': 'No valid correlations calculated'}
        
        return {
            'overall_correlation': {
                'mean': float(np.mean(correlations)),
                'std': float(np.std(correlations)),
                'min': float(np.min(correlations)),
                'max': float(np.max(correlations))
            },
            'high_performance_high_believability': len([c for c in correlations if c > 0.5]),
            'low_correlation': len([c for c in correlations if abs(c) < 0.2]),
            'negative_correlation': len([c for c in correlations if c < -0.2])
        }
    
    def save_enhanced_believability_results(self, results):
        """Save enhanced believability results"""
        
        # Save enhanced JSON results
        enhanced_file = os.path.join(
            self.results_dir, 
            f"{self.tournament_name}_enhanced_believability.json"
        )
        
        with open(enhanced_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save human-readable believability report
        self._save_detailed_believability_report(results)
        
        print(f"Enhanced believability results saved to {enhanced_file}")
    
    def _save_detailed_believability_report(self, results):
        """Save detailed human-readable believability report"""
        report_file = os.path.join(
            self.results_dir, 
            f"{self.tournament_name}_detailed_believability_report.txt"
        )
        
        with open(report_file, 'w') as f:
            f.write(f"ENHANCED BELIEVABILITY ANALYSIS REPORT\n")
            f.write(f"Tournament: {self.tournament_name}\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            analysis = results.believability_analysis
            f.write(f"Total Enhanced Clues Analyzed: {analysis.get('enhanced_metrics_count', 0)}\n")
            f.write(f"Agents with Believability Profiles: {len(self.agent_believability_profiles)}\n\n")
            
            # Metric distributions
            distributions = analysis.get('metric_distributions', {})
            if distributions:
                f.write("METRIC DISTRIBUTIONS\n")
                f.write("-" * 40 + "\n")
                for metric, stats in distributions.items():
                    f.write(f"\n{metric.upper()}:\n")
                    f.write(f"  Mean: {stats['mean']:.3f}\n")
                    f.write(f"  Std Dev: {stats['std']:.3f}\n")
                    f.write(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]\n")
                    f.write(f"  Median: {stats['median']:.3f}\n")
                    f.write(f"  Q25-Q75: [{stats['q25']:.3f}, {stats['q75']:.3f}]\n")
            
            # Top believable clues
            top_clues = analysis.get('top_believable_clues', [])
            if top_clues:
                f.write(f"\n\nTOP {len(top_clues)} MOST BELIEVABLE CLUES\n")
                f.write("-" * 40 + "\n")
                for i, clue_data in enumerate(top_clues, 1):
                    f.write(f"\n{i}. \"{clue_data['clue']}\" by {clue_data['codemaster']}\n")
                    f.write(f"   Targets: {', '.join(clue_data['targets'])}\n")
                    f.write(f"   Composite Believability: {clue_data['composite_believability']:.3f}\n")
                    f.write(f"   Frequency: {clue_data['frequency_score']:.3f}, ")
                    f.write(f"Coherence: {clue_data['coherence_score']:.3f}, ")
                    f.write(f"Human-like: {clue_data['human_likeness']:.3f}\n")
            
            # Agent believability profiles
            f.write(f"\n\nAGENT BELIEVABILITY PROFILES\n")
            f.write("-" * 40 + "\n")
            
            # Sort agents by composite believability
            sorted_profiles = sorted(
                self.agent_believability_profiles.items(),
                key=lambda x: x[1].composite_believability,
                reverse=True
            )
            
            for agent_name, profile in sorted_profiles:
                f.write(f"\n{agent_name}:\n")
                f.write(f"  Composite Believability: {profile.composite_believability:.3f}\n")
                f.write(f"  Component Scores:\n")
                f.write(f"    Frequency: {profile.avg_frequency_score:.3f}\n")
                f.write(f"    Semantic Coherence: {profile.avg_semantic_coherence:.3f}\n")
                f.write(f"    Human-likeness: {profile.avg_human_likeness:.3f}\n")
                f.write(f"    Diversity: {profile.diversity_score:.3f}\n")
                f.write(f"    Safety: {profile.avg_safety_score:.3f}\n")
                f.write(f"  Performance-Believability Correlation: {profile.performance_believability_correlation:.3f}\n")
                f.write(f"  Total Clues Analyzed: {len(profile.clue_metrics)}\n")
                
                # Z-scores
                if profile.z_normalized_scores:
                    f.write(f"  Z-scores vs Human Baseline:\n")
                    for metric, zscore in profile.z_normalized_scores.items():
                        f.write(f"    {metric}: {zscore:.2f}\n")
            
            # Performance-believability analysis
            perf_analysis = analysis.get('performance_believability_analysis', {})
            if 'overall_correlation' in perf_analysis:
                f.write(f"\n\nPERFORMANCE-BELIEVABILITY CORRELATION ANALYSIS\n")
                f.write("-" * 50 + "\n")
                overall = perf_analysis['overall_correlation']
                f.write(f"Overall Correlation Statistics:\n")
                f.write(f"  Mean Correlation: {overall['mean']:.3f}\n")
                f.write(f"  Standard Deviation: {overall['std']:.3f}\n")
                f.write(f"  Range: [{overall['min']:.3f}, {overall['max']:.3f}]\n")
                f.write(f"\nCorrelation Categories:\n")
                f.write(f"  High Performance + High Believability: {perf_analysis.get('high_performance_high_believability', 0)} agents\n")
                f.write(f"  Low Correlation (|r| < 0.2): {perf_analysis.get('low_correlation', 0)} agents\n")
                f.write(f"  Negative Correlation (r < -0.2): {perf_analysis.get('negative_correlation', 0)} agents\n")
            
            # Recommendations
            f.write(f"\n\nRECOMMENDations FOR IMPROVEMENT\n")
            f.write("-" * 40 + "\n")
            
            # Find agents with low believability
            low_believability = [
                (name, profile) for name, profile in sorted_profiles
                if profile.composite_believability < 0.4
            ]
            
            if low_believability:
                f.write("Agents with Low Believability Scores:\n")
                for name, profile in low_believability:
                    f.write(f"  {name}: {profile.composite_believability:.3f}\n")
                    
                    # Specific recommendations
                    recommendations = []
                    if profile.avg_frequency_score < 0.3:
                        recommendations.append("Use more common/typical words")
                    if profile.avg_semantic_coherence < 0.4:
                        recommendations.append("Improve semantic connections to targets")
                    if profile.avg_human_likeness < 0.4:
                        recommendations.append("Use more human-like word patterns")
                    if profile.diversity_score < 0.5:
                        recommendations.append("Increase clue diversity")
                    if profile.avg_safety_score < 0.8:
                        recommendations.append("Avoid potentially offensive language")
                    
                    if recommendations:
                        f.write(f"    Recommendations: {'; '.join(recommendations)}\n")
            
            # High-performing but low-believability agents
            high_perf_low_believe = []
            for name, profile in self.agent_believability_profiles.items():
                if name in self.agent_metrics:
                    agent_perf = self.agent_metrics[name]
                    win_rate = agent_perf.wins / max(1, agent_perf.games_played)
                    if win_rate > 0.6 and profile.composite_believability < 0.5:
                        high_perf_low_believe.append((name, win_rate, profile.composite_believability))
            
            if high_perf_low_believe:
                f.write(f"\nHigh-Performance but Low-Believability Agents:\n")
                for name, win_rate, believability in high_perf_low_believe:
                    f.write(f"  {name}: {win_rate:.1%} win rate, {believability:.3f} believability\n")
                f.write("These agents might benefit from believability-focused training.\n")
        
        print(f"Detailed believability report saved to {report_file}")

# Convenience function for easy import
def run_enhanced_believability_tournament(
    tournament_name: str = "Enhanced_Believability_Championship",
    games_per_matchup: int = 2,
    max_matchups: int = 300
) -> EnhancedBelievabilityTournament:
    """
    Create and return an enhanced believability tournament instance
    
    Args:
        tournament_name: Name of the tournament
        games_per_matchup: Number of games each pair of teams plays
        max_matchups: Maximum number of matchups to run
    
    Returns:
        EnhancedBelievabilityTournament instance ready for agent registration
    """
    
    tournament = EnhancedBelievabilityTournament(
        tournament_name=tournament_name,
        games_per_matchup=games_per_matchup,
        max_matchups=max_matchups
    )
    
    print(f"Enhanced Believability Tournament '{tournament_name}' created")
    print(f"Configuration: {games_per_matchup} games per matchup, max {max_matchups} matchups")
    print("Ready for agent registration via tournament.register_agent()")
    
    return tournament

# Backwards compatibility
BelievabilityTournament = EnhancedBelievabilityTournament