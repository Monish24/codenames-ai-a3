import json
import random
import time
import os
import itertools
import math
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
import trueskill
import numpy as np
import scipy.stats as stats

from game import Game
from players.codemaster import *
from players.guesser import *

@dataclass
class Agent:
    name: str
    agent_type: str  # 'codemaster' or 'guesser'
    class_reference: Any
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}

@dataclass
class ClueRecord:
    """Record of a single clue given during gameplay"""
    clue_word: str
    clue_number: int
    codemaster_name: str
    target_words: List[str]
    guessed_words: List[str]
    correct_guesses: List[str]
    turn_number: int
    team_color: str
    game_seed: int
    timestamp: str
    revealed_cards: int = 0  # Total cards revealed this turn
    target_cards: int = 0    # Number of target cards revealed
    
    def __post_init__(self):
        self.target_cards = len([w for w in self.guessed_words if w in self.target_words])
        self.revealed_cards = len(self.guessed_words)

@dataclass
class GuessRecord:
    """Record of a single guess made during gameplay"""
    guess_word: str
    guesser_name: str
    clue_word: str
    clue_number: int
    is_correct: bool
    card_type: str  # 'team', 'opponent', 'civilian', 'assassin'
    confidence: float  # If available from guesser
    turn_number: int
    guess_order: int  # Order of this guess within the turn
    team_color: str
    game_seed: int
    timestamp: str

@dataclass
class MatchResult:
    match_id: str
    red_codemaster: str
    red_guesser: str
    blue_codemaster: str
    blue_guesser: str
    winner: str  # 'R' or 'B'
    total_turns: int
    red_words_found: int
    blue_words_found: int
    civilians_hit: int
    assassin_hit: bool
    game_duration: float
    seed: int
    timestamp: str
    # Enhanced metrics
    red_turns_to_victory: Optional[int] = None
    blue_turns_to_victory: Optional[int] = None
    clue_records: List[ClueRecord] = field(default_factory=list)
    guess_records: List[GuessRecord] = field(default_factory=list)

@dataclass
class AgentMetrics:
    """Comprehensive metrics for individual agents"""
    name: str
    agent_type: str  # 'codemaster' or 'guesser'
    
    # Basic performance
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    
    # TrueSkill ratings
    trueskill_rating: Any = None
    role_based_rating: Any = None  # For decoupled role ranking
    
    # Performance metrics
    avg_turns_to_victory: float = 0.0
    avg_turns_all_games: float = 0.0
    
    # Role-specific metrics
    codemaster_metrics: Dict[str, float] = field(default_factory=dict)
    guesser_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Believability scores
    believability_scores: Dict[str, float] = field(default_factory=dict)
    composite_believability: float = 0.0
    
    def __post_init__(self):
        if self.trueskill_rating is None:
            self.trueskill_rating = trueskill.Rating()
        if self.role_based_rating is None:
            self.role_based_rating = trueskill.Rating()

@dataclass
class TeamResult:
    codemaster: str
    guesser: str
    wins: int = 0
    losses: int = 0
    total_games: int = 0
    avg_turns_when_winning: float = 0.0
    avg_words_found: float = 0.0
    trueskill_rating: Any = None
    
    # Enhanced team metrics
    avg_turns_all_games: float = 0.0
    team_synergy_score: float = 0.0
    consistency_score: float = 0.0  # Variance in performance
    
    def __post_init__(self):
        if self.trueskill_rating is None:
            self.trueskill_rating = trueskill.Rating()
    
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.total_games)
    
    @property
    def conservative_skill(self) -> float:
        """Conservative skill estimate: μ - 3σ"""
        return self.trueskill_rating.mu - 3 * self.trueskill_rating.sigma
    
    @property
    def wilson_confidence_interval(self) -> Tuple[float, float]:
        """Wilson 95% confidence interval for win rate"""
        if self.total_games == 0:
            return (0.0, 0.0)
        
        n = self.total_games
        p = self.win_rate
        z = 1.96  # 95% confidence
        
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator
        
        return (max(0, center - margin), min(1, center + margin))

class BelievabilityAnalyzer:
    """Analyzes clue and guess believability using multiple metrics"""
    
    def __init__(self):
        self.word_frequencies = self._load_word_frequencies()
        self.human_baseline_stats = self._load_human_baseline()
        
        # Initialize models for believability analysis
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded SentenceTransformer for believability analysis")
        except:
            self.sentence_model = None
            print("SentenceTransformer not available for believability analysis")
    
    def _load_word_frequencies(self) -> Dict[str, float]:
        """Load or create word frequency model"""
        # In a real implementation, load from human gameplay corpus
        # For now, use a simple model based on word length and common patterns
        frequencies = {}
        
        # Add some common words with high frequencies
        common_words = [
            'animal', 'food', 'color', 'number', 'big', 'small', 'hot', 'cold',
            'water', 'fire', 'earth', 'air', 'game', 'sport', 'music', 'art',
            'home', 'work', 'school', 'nature', 'science', 'history'
        ]
        
        for word in common_words:
            frequencies[word] = 0.01  # High frequency
        
        return frequencies
    
    def _load_human_baseline(self) -> Dict[str, Dict[str, float]]:
        """Load human baseline statistics for z-score normalization"""
        # In real implementation, load from human gameplay analysis
        return {
            'frequency': {'mean': 0.005, 'std': 0.003},
            'coherence': {'mean': 0.6, 'std': 0.2},
            'human_likeness': {'mean': 0.7, 'std': 0.15},
            'diversity': {'mean': 0.8, 'std': 0.1},
            'safety': {'mean': 0.95, 'std': 0.05}
        }
    
    def calculate_frequency_score(self, clue: str) -> float:
        """Calculate frequency/typicality score (lower surprisal = more typical)"""
        clue_lower = clue.lower()
        
        # Get base frequency
        base_freq = self.word_frequencies.get(clue_lower, 0.0001)
        
        # Adjust for word characteristics
        length_penalty = 0.1 if len(clue) > 10 else 0
        unusual_pattern_penalty = 0.2 if any(pattern in clue_lower 
                                           for pattern in ['ph', 'gh', 'sch']) else 0
        
        # Convert to surprisal (negative log probability)
        surprisal = -math.log(base_freq + 1e-10)
        
        # Normalize to 0-1 scale (lower surprisal = higher score)
        normalized_score = max(0, min(1, 1 - (surprisal / 10)))
        
        return max(0, normalized_score - length_penalty - unusual_pattern_penalty)
    
    def calculate_semantic_coherence(self, clue: str, targets: List[str], 
                                   distractors: List[str]) -> float:
        """Calculate semantic coherence: target_sim - distractor_sim"""
        if not self.sentence_model or not targets:
            return 0.5  # Neutral score if can't calculate
        
        try:
            # Encode all words
            all_words = [clue] + targets + distractors
            embeddings = self.sentence_model.encode(all_words)
            
            clue_embedding = embeddings[0]
            target_embeddings = embeddings[1:len(targets)+1]
            distractor_embeddings = embeddings[len(targets)+1:]
            
            # Calculate similarities
            target_sims = [np.dot(clue_embedding, t_emb) / 
                          (np.linalg.norm(clue_embedding) * np.linalg.norm(t_emb))
                          for t_emb in target_embeddings]
            
            distractor_sims = [np.dot(clue_embedding, d_emb) / 
                              (np.linalg.norm(clue_embedding) * np.linalg.norm(d_emb))
                              for d_emb in distractor_embeddings] if distractor_embeddings.size > 0 else [0]
            
            avg_target_sim = np.mean(target_sims) if target_sims else 0
            avg_distractor_sim = np.mean(distractor_sims) if distractor_sims else 0
            
            # Normalize to 0-1 scale
            coherence = (avg_target_sim - avg_distractor_sim + 1) / 2
            return max(0, min(1, coherence))
            
        except Exception as e:
            print(f"Error calculating semantic coherence: {e}")
            return 0.5
    
    def calculate_human_likeness(self, clue: str) -> float:
        """Calculate how human-like a clue appears"""
        score = 0.5  # Base score
        
        # Length preferences (4-8 characters optimal for humans)
        if 4 <= len(clue) <= 8:
            score += 0.2
        elif len(clue) < 4 or len(clue) > 12:
            score -= 0.2
        
        # Common word patterns humans use
        if any(clue.lower().endswith(suffix) for suffix in 
               ['ing', 'tion', 'ness', 'ment', 'able', 'ful']):
            score += 0.15
        
        # Avoid overly technical or obscure patterns
        if any(pattern in clue.lower() for pattern in 
               ['ph', 'sch', 'chr', 'ps', 'gn']):
            score -= 0.2
        
        # Prefer common letter combinations
        vowel_ratio = sum(1 for c in clue.lower() if c in 'aeiou') / len(clue)
        if 0.2 <= vowel_ratio <= 0.5:
            score += 0.1
        
        return max(0, min(1, score))
    
    def calculate_diversity_score(self, agent_clues: List[str]) -> float:
        """Calculate diversity using type/token ratio and self-BLEU"""
        if not agent_clues:
            return 1.0
        
        # Type/token ratio
        unique_clues = len(set(clue.lower() for clue in agent_clues))
        total_clues = len(agent_clues)
        type_token_ratio = unique_clues / total_clues
        
        # Simple self-similarity penalty (poor man's self-BLEU)
        clue_counter = Counter(clue.lower() for clue in agent_clues)
        max_repetition = max(clue_counter.values()) if clue_counter else 1
        repetition_penalty = max_repetition / total_clues
        
        diversity = type_token_ratio * (1 - repetition_penalty + 0.1)
        return max(0, min(1, diversity))
    
    def calculate_safety_score(self, clue: str) -> float:
        """Calculate safety/toxicity score (1.0 = safe, 0.0 = toxic)"""
        # Simple heuristic-based safety check
        # In real implementation, use Perspective API or similar
        
        unsafe_patterns = [
            'hate', 'kill', 'die', 'dead', 'blood', 'war', 'bomb', 'gun',
            'stupid', 'idiot', 'damn', 'hell'
        ]
        
        clue_lower = clue.lower()
        safety_score = 1.0
        
        for pattern in unsafe_patterns:
            if pattern in clue_lower:
                safety_score -= 0.3
        
        # Check for excessive caps (shouting)
        if clue.isupper() and len(clue) > 3:
            safety_score -= 0.1
        
        return max(0, min(1, safety_score))
    
    def calculate_composite_believability(self, clue_record: ClueRecord,
                                        agent_clues: List[str],
                                        board_words: List[str]) -> Dict[str, float]:
        """Calculate composite believability score for a clue"""
        
        clue = clue_record.clue_word
        targets = clue_record.target_words
        
        # Get distractors (non-target board words)
        distractors = [w for w in board_words if w not in targets]
        
        # Calculate individual metrics
        frequency = self.calculate_frequency_score(clue)
        coherence = self.calculate_semantic_coherence(clue, targets, distractors)
        human_likeness = self.calculate_human_likeness(clue)
        diversity = self.calculate_diversity_score(agent_clues)
        safety = self.calculate_safety_score(clue)
        
        # Calculate z-scores against human baseline
        metrics = {
            'frequency': frequency,
            'coherence': coherence,
            'human_likeness': human_likeness,
            'diversity': diversity,
            'safety': safety
        }
        
        z_scores = {}
        for metric, value in metrics.items():
            baseline = self.human_baseline_stats.get(metric, {'mean': 0.5, 'std': 0.1})
            z_score = (value - baseline['mean']) / baseline['std']
            z_scores[f'{metric}_zscore'] = z_score
        
        # Composite score with weights
        composite = (0.35 * coherence + 
                    0.20 * frequency + 
                    0.20 * human_likeness + 
                    0.15 * diversity + 
                    0.10 * safety)
        
        return {
            **metrics,
            **z_scores,
            'composite_believability': composite
        }

@dataclass 
class EnhancedTournamentResults:
    """Comprehensive tournament results with all metrics"""
    tournament_name: str
    total_games: int
    match_results: List[MatchResult]
    team_rankings: List[Tuple[str, TeamResult]]
    agent_rankings: List[Tuple[str, AgentMetrics]]
    codemaster_rankings: List[Tuple[str, AgentMetrics]]
    guesser_rankings: List[Tuple[str, AgentMetrics]]
    believability_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'tournament_name': self.tournament_name,
            'total_games': self.total_games,
            'match_results': [asdict(r) for r in self.match_results],
            'team_rankings': [(k, asdict(v)) for k, v in self.team_rankings],
            'agent_rankings': [(k, asdict(v)) for k, v in self.agent_rankings],
            'codemaster_rankings': [(k, asdict(v)) for k, v in self.codemaster_rankings],
            'guesser_rankings': [(k, asdict(v)) for k, v in self.guesser_rankings],
            'believability_analysis': self.believability_analysis
        }

class EnhancedTournamentManager:
    """Enhanced tournament manager with comprehensive metrics and believability analysis"""
    
    def __init__(self, 
                 tournament_name: str = "Enhanced_Codenames_Tournament",
                 results_dir: str = "tournament_results",
                 games_per_matchup: int = 1,
                 max_matchups: int = 500):
        
        # Pre-load shared models
        from model_manager import preload_models, get_model_info
        preload_models()
        info = get_model_info()
        print(f"📊 Loaded {info['model_count']} models: {info['loaded_models']}")
        
        self.tournament_name = tournament_name
        self.results_dir = results_dir
        self.games_per_matchup = games_per_matchup
        self.max_matchups = max_matchups
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize data structures
        self.codemasters: List[Agent] = []
        self.guessers: List[Agent] = []
        self.match_results: List[MatchResult] = []
        self.team_results: Dict[str, TeamResult] = {}
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
        # TrueSkill environments
        self.team_ts_env = trueskill.TrueSkill()
        self.role_ts_env = trueskill.TrueSkill()  # For role-based ranking
        
        # Believability analyzer
        self.believability_analyzer = BelievabilityAnalyzer()
        
        # Data collection
        self.all_clue_records: List[ClueRecord] = []
        self.all_guess_records: List[GuessRecord] = []
    
    def register_agent(self, name: str, agent_type: str, class_reference: Any, **kwargs):
        """Register an agent for the tournament"""
        agent = Agent(name, agent_type, class_reference, kwargs)
        
        if agent_type == 'codemaster':
            self.codemasters.append(agent)
        elif agent_type == 'guesser':
            self.guessers.append(agent)
        else:
            raise ValueError(f"Invalid agent_type: {agent_type}")
        
        # Initialize agent metrics
        self.agent_metrics[name] = AgentMetrics(name=name, agent_type=agent_type)
        
        print(f"Registered {agent_type}: {name}")
    
    def generate_team_combinations(self) -> List[Tuple[Agent, Agent]]:
        """Generate all possible team combinations"""
        teams = []
        for cm in self.codemasters:
            for g in self.guessers:
                teams.append((cm, g))
                # Initialize team result
                team_key = f"{cm.name}+{g.name}"
                if team_key not in self.team_results:
                    self.team_results[team_key] = TeamResult(
                        codemaster=cm.name, 
                        guesser=g.name
                    )
        return teams
    
    def generate_matchups(self) -> List[Tuple[Tuple[Agent, Agent], Tuple[Agent, Agent]]]:
        """Generate matchups with intelligent reduction if needed"""
        teams = self.generate_team_combinations()
        all_matchups = []
        
        print(f"Total teams: {len(teams)}")
        
        # Generate all possible matchups
        for i, team_a in enumerate(teams):
            for j, team_b in enumerate(teams):
                if i != j:  # Teams don't play against themselves
                    all_matchups.append((team_a, team_b))
        
        print(f"Total possible matchups: {len(all_matchups)}")
        
        # Intelligent reduction if needed
        if len(all_matchups) > self.max_matchups:
            print(f"Reducing from {len(all_matchups)} to {self.max_matchups} matchups...")
            final_matchups = self._select_balanced_matchups(all_matchups, teams)
            print(f"Final matchups selected: {len(final_matchups)}")
            return final_matchups
        else:
            print(f"Using all {len(all_matchups)} matchups")
            return all_matchups
    
    def _select_balanced_matchups(self, all_matchups: List, teams: List) -> List:
        """Select balanced matchups ensuring each team plays minimum games"""
        final_matchups = []
        team_game_count = defaultdict(int)
        
        # Shuffle for randomness
        random.shuffle(all_matchups)
        
        # Ensure each team plays at least 3 games
        min_games_per_team = 3
        for matchup in all_matchups:
            team_a, team_b = matchup
            team_a_key = f"{team_a[0].name}+{team_a[1].name}"
            team_b_key = f"{team_b[0].name}+{team_b[1].name}"
            
            if (team_game_count[team_a_key] < min_games_per_team or 
                team_game_count[team_b_key] < min_games_per_team):
                final_matchups.append(matchup)
                team_game_count[team_a_key] += 1
                team_game_count[team_b_key] += 1
                
                if len(final_matchups) >= self.max_matchups:
                    break
        
        # Fill remaining slots
        remaining_matchups = [m for m in all_matchups if m not in final_matchups]
        random.shuffle(remaining_matchups)
        
        slots_remaining = self.max_matchups - len(final_matchups)
        final_matchups.extend(remaining_matchups[:slots_remaining])
        
        return final_matchups
    
    def run_single_game(self, red_team: Tuple[Agent, Agent], 
                       blue_team: Tuple[Agent, Agent],
                       game_seed: int = None) -> MatchResult:
        """Run a single game and collect comprehensive metrics"""
        
        red_cm, red_g = red_team
        blue_cm, blue_g = blue_team
        
        if game_seed is None:
            game_seed = random.randint(1, 1000000)
        
        match_id = f"{red_cm.name}+{red_g.name}_vs_{blue_cm.name}+{blue_g.name}_{game_seed}"
        
        print(f"Running match: {match_id}")
        
        start_time = time.time()
        
        try:
            # Create game instance with enhanced logging
            game = Game(
                codemaster_red=red_cm.class_reference,
                guesser_red=red_g.class_reference,
                codemaster_blue=blue_cm.class_reference,
                guesser_blue=blue_g.class_reference,
                seed=game_seed,
                do_print=True,
                do_log=False,
                cmr_kwargs=red_cm.kwargs,
                gr_kwargs=red_g.kwargs,
                cmb_kwargs=blue_cm.kwargs,
                gb_kwargs=blue_g.kwargs
            )
            
            # Run the game
            game.run()
            
            end_time = time.time()
            
            # Extract basic results
            red_words = game.words_on_board.count("*Red*")
            blue_words = game.words_on_board.count("*Blue*")
            civilians = game.words_on_board.count("*Civilian*")
            assassin = game.words_on_board.count("*Assassin*") > 0
            
            # Calculate turns to victory
            red_turns = None
            blue_turns = None
            if game.game_winner == 'R':
                red_turns = getattr(game, 'total_turns', 0)
            elif game.game_winner == 'B':
                blue_turns = getattr(game, 'total_turns', 0)
            
            # Create comprehensive result
            result = MatchResult(
                match_id=match_id,
                red_codemaster=red_cm.name,
                red_guesser=red_g.name,
                blue_codemaster=blue_cm.name,
                blue_guesser=blue_g.name,
                winner=game.game_winner,
                total_turns=getattr(game, 'total_turns', 0),
                red_words_found=red_words,
                blue_words_found=blue_words,
                civilians_hit=civilians,
                assassin_hit=assassin,
                game_duration=end_time - start_time,
                seed=game_seed,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                red_turns_to_victory=red_turns,
                blue_turns_to_victory=blue_turns
            )
            
            return result
            
        except Exception as e:
            print(f"Error in match {match_id}: {str(e)}")
            return MatchResult(
                match_id=match_id,
                red_codemaster=red_cm.name,
                red_guesser=red_g.name,
                blue_codemaster=blue_cm.name,
                blue_guesser=blue_g.name,
                winner="ERROR",
                total_turns=0,
                red_words_found=0,
                blue_words_found=0,
                civilians_hit=0,
                assassin_hit=False,
                game_duration=0,
                seed=game_seed,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    def update_comprehensive_metrics(self, result: MatchResult):
        """Update all metrics based on game results"""
        if result.winner == "ERROR":
            return
        
        # Update team metrics
        self._update_team_metrics(result)
        
        # Update individual agent metrics
        self._update_agent_metrics(result)
    
    def _update_team_metrics(self, result: MatchResult):
        """Update team-level metrics"""
        red_team_key = f"{result.red_codemaster}+{result.red_guesser}"
        blue_team_key = f"{result.blue_codemaster}+{result.blue_guesser}"
        
        red_stats = self.team_results[red_team_key]
        blue_stats = self.team_results[blue_team_key]
        
        # Update win/loss records
        if result.winner == 'R':
            red_stats.wins += 1
            blue_stats.losses += 1
            if result.red_turns_to_victory:
                red_stats.avg_turns_when_winning = (
                    (red_stats.avg_turns_when_winning * (red_stats.wins - 1) + 
                     result.red_turns_to_victory) / red_stats.wins
                )
        elif result.winner == 'B':
            blue_stats.wins += 1
            red_stats.losses += 1
            if result.blue_turns_to_victory:
                blue_stats.avg_turns_when_winning = (
                    (blue_stats.avg_turns_when_winning * (blue_stats.wins - 1) + 
                     result.blue_turns_to_victory) / blue_stats.wins
                )
        
        # Update game counts and averages
        for stats, words_found in [(red_stats, result.red_words_found), 
                                  (blue_stats, result.blue_words_found)]:
            stats.total_games += 1
            stats.avg_words_found = (
                (stats.avg_words_found * (stats.total_games - 1) + words_found) / 
                stats.total_games
            )
            stats.avg_turns_all_games = (
                (stats.avg_turns_all_games * (stats.total_games - 1) + result.total_turns) / 
                stats.total_games
            )
    
    def _update_agent_metrics(self, result: MatchResult):
        """Update individual agent metrics"""
        agents = [
            (result.red_codemaster, result.winner == 'R'),
            (result.red_guesser, result.winner == 'R'),
            (result.blue_codemaster, result.winner == 'B'),
            (result.blue_guesser, result.winner == 'B')
        ]
        
        for agent_name, won in agents:
            if agent_name in self.agent_metrics:
                metrics = self.agent_metrics[agent_name]
                metrics.games_played += 1
                
                if won:
                    metrics.wins += 1
                    # Update turns to victory
                    turns = (result.red_turns_to_victory if agent_name in 
                            [result.red_codemaster, result.red_guesser] 
                            else result.blue_turns_to_victory)
                    if turns:
                        metrics.avg_turns_to_victory = (
                            (metrics.avg_turns_to_victory * (metrics.wins - 1) + turns) / 
                            metrics.wins
                        )
                else:
                    metrics.losses += 1
                
                # Update overall turn average
                metrics.avg_turns_all_games = (
                    (metrics.avg_turns_all_games * (metrics.games_played - 1) + 
                     result.total_turns) / metrics.games_played
                )
    
    def calculate_all_trueskill_ratings(self):
        """Calculate both team-based and role-based TrueSkill ratings"""
        print("Calculating TrueSkill ratings...")
        
        # Initialize team ratings
        for team_key in self.team_results:
            self.team_results[team_key].trueskill_rating = trueskill.Rating()
        
        # Initialize role-based ratings for agents
        for agent_name in self.agent_metrics:
            self.agent_metrics[agent_name].role_based_rating = trueskill.Rating()
        
        # Process each match for team-based TrueSkill
        for result in self.match_results:
            if result.winner in ['R', 'B']:
                self._update_team_trueskill(result)
                self._update_role_based_trueskill(result)
    
    def _update_team_trueskill(self, result: MatchResult):
        """Update team-based TrueSkill ratings"""
        red_team_key = f"{result.red_codemaster}+{result.red_guesser}"
        blue_team_key = f"{result.blue_codemaster}+{result.blue_guesser}"
        
        red_rating = self.team_results[red_team_key].trueskill_rating
        blue_rating = self.team_results[blue_team_key].trueskill_rating
        
        # Determine ranks (0 = winner, 1 = loser)
        if result.winner == 'R':
            ranks = [0, 1]  # Red wins
        else:  # result.winner == 'B'
            ranks = [1, 0]  # Blue wins
        
        # Update ratings
        new_red, new_blue = trueskill.rate([(red_rating,), (blue_rating,)], ranks=ranks)
        
        self.team_results[red_team_key].trueskill_rating = new_red[0]
        self.team_results[blue_team_key].trueskill_rating = new_blue[0]
    
    def _update_role_based_trueskill(self, result: MatchResult):
        """Update role-based TrueSkill ratings (codemasters and guessers separately)"""
        # Get agent ratings
        red_cm_rating = self.agent_metrics[result.red_codemaster].role_based_rating
        red_g_rating = self.agent_metrics[result.red_guesser].role_based_rating
        blue_cm_rating = self.agent_metrics[result.blue_codemaster].role_based_rating
        blue_g_rating = self.agent_metrics[result.blue_guesser].role_based_rating
        
        # Create team ratings for role-based calculation
        # Treat this as 4-player game with 2-person teams
        red_team_rating = [red_cm_rating, red_g_rating]
        blue_team_rating = [blue_cm_rating, blue_g_rating]
        
        # Determine ranks
        if result.winner == 'R':
            ranks = [0, 0, 1, 1]  # Red team wins
        else:  # result.winner == 'B'
            ranks = [1, 1, 0, 0]  # Blue team wins
        
        # Update ratings
        new_ratings = trueskill.rate([
            (red_cm_rating,), (red_g_rating,), 
            (blue_cm_rating,), (blue_g_rating,)
        ], ranks=ranks)
        
        # Store updated ratings
        self.agent_metrics[result.red_codemaster].role_based_rating = new_ratings[0][0]
        self.agent_metrics[result.red_guesser].role_based_rating = new_ratings[1][0]
        self.agent_metrics[result.blue_codemaster].role_based_rating = new_ratings[2][0]
        self.agent_metrics[result.blue_guesser].role_based_rating = new_ratings[3][0]
    
    def calculate_role_specific_metrics(self):
        """Calculate role-specific performance metrics"""
        print("Calculating role-specific metrics...")
        
        # Group records by agent
        agent_clue_records = defaultdict(list)
        agent_guess_records = defaultdict(list)
        
        for clue_record in self.all_clue_records:
            agent_clue_records[clue_record.codemaster_name].append(clue_record)
        
        for guess_record in self.all_guess_records:
            agent_guess_records[guess_record.guesser_name].append(guess_record)
        
        # Calculate codemaster metrics
        for agent_name, clue_records in agent_clue_records.items():
            if agent_name in self.agent_metrics:
                metrics = self._calculate_codemaster_metrics(clue_records)
                self.agent_metrics[agent_name].codemaster_metrics = metrics
        
        # Calculate guesser metrics
        for agent_name, guess_records in agent_guess_records.items():
            if agent_name in self.agent_metrics:
                metrics = self._calculate_guesser_metrics(guess_records)
                self.agent_metrics[agent_name].guesser_metrics = metrics
    
    def _calculate_codemaster_metrics(self, clue_records: List[ClueRecord]) -> Dict[str, float]:
        """Calculate codemaster-specific metrics"""
        if not clue_records:
            return {}
        
        # Clue efficiency: target_cards / revealed_cards
        efficiencies = []
        for record in clue_records:
            if record.revealed_cards > 0:
                efficiency = record.target_cards / record.revealed_cards
                efficiencies.append(efficiency)
        
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        
        # One-step solvability: % of clues where all targets were found
        perfect_clues = sum(1 for r in clue_records 
                           if r.target_cards == len(r.target_words) and r.target_cards > 0)
        solvability = perfect_clues / len(clue_records) if clue_records else 0.0
        
        # Average clue connectivity (how many words typically targeted)
        avg_targets = np.mean([len(r.target_words) for r in clue_records])
        
        # Risk management (avoiding assassin/opponent hits)
        safe_clues = sum(1 for r in clue_records 
                        if not any(word.startswith("*Assassin*") or 
                                 word.startswith("*Blue*") or word.startswith("*Red*") 
                                 for word in r.guessed_words 
                                 if word not in r.target_words))
        safety_rate = safe_clues / len(clue_records) if clue_records else 1.0
        
        return {
            'avg_clue_efficiency': avg_efficiency,
            'one_step_solvability': solvability,
            'avg_targets_per_clue': avg_targets,
            'safety_rate': safety_rate,
            'total_clues_given': len(clue_records)
        }
    
    def _calculate_guesser_metrics(self, guess_records: List[GuessRecord]) -> Dict[str, float]:
        """Calculate guesser-specific metrics"""
        if not guess_records:
            return {}
        
        # Guess accuracy: correct_guesses / total_guesses
        correct_guesses = sum(1 for r in guess_records if r.is_correct)
        accuracy = correct_guesses / len(guess_records)
        
        # First guess accuracy (how often first guess in turn is correct)
        first_guesses = [r for r in guess_records if r.guess_order == 1]
        first_guess_accuracy = (sum(1 for r in first_guesses if r.is_correct) / 
                               len(first_guesses) if first_guesses else 0.0)
        
        # Risk aversion (how often they hit dangerous cards)
        dangerous_hits = sum(1 for r in guess_records 
                            if r.card_type in ['assassin', 'opponent'])
        risk_score = 1.0 - (dangerous_hits / len(guess_records))
        
        # Clue following ability (how well they stick to clue intent)
        # This is approximated by looking at guess patterns
        multi_guess_turns = defaultdict(list)
        for record in guess_records:
            turn_key = f"{record.game_seed}_{record.turn_number}"
            multi_guess_turns[turn_key].append(record)
        
        consistent_turns = 0
        total_multi_turns = 0
        for turn_guesses in multi_guess_turns.values():
            if len(turn_guesses) > 1:
                total_multi_turns += 1
                # Check if guesses were consistent (similar card types)
                card_types = [g.card_type for g in turn_guesses]
                if len(set(card_types)) <= 2:  # At most 2 different types
                    consistent_turns += 1
        
        consistency = (consistent_turns / total_multi_turns 
                      if total_multi_turns > 0 else 1.0)
        
        return {
            'guess_accuracy': accuracy,
            'first_guess_accuracy': first_guess_accuracy,
            'risk_management': risk_score,
            'guess_consistency': consistency,
            'total_guesses_made': len(guess_records)
        }
    
    def analyze_believability(self):
        """Perform comprehensive believability analysis"""
        print("Analyzing clue believability...")
        
        # Group clues by agent
        agent_clues = defaultdict(list)
        for record in self.all_clue_records:
            agent_clues[record.codemaster_name].append(record)
        
        # Analyze each agent's believability
        for agent_name, clue_records in agent_clues.items():
            if agent_name in self.agent_metrics:
                believability_scores = self._analyze_agent_believability(
                    clue_records, agent_name
                )
                self.agent_metrics[agent_name].believability_scores = believability_scores
                self.agent_metrics[agent_name].composite_believability = (
                    believability_scores.get('composite_believability', 0.5)
                )
    
    def _analyze_agent_believability(self, clue_records: List[ClueRecord], 
                                   agent_name: str) -> Dict[str, float]:
        """Analyze believability for a specific agent"""
        if not clue_records:
            return {'composite_believability': 0.5}
        
        # Get all clues for diversity calculation
        agent_clues = [r.clue_word for r in clue_records]
        
        # Calculate believability for each clue
        clue_believabilities = []
        for record in clue_records:
            # Get board words for this game (approximate)
            board_words = record.target_words + ['dummy'] * 20  # Simplified
            
            believability = self.believability_analyzer.calculate_composite_believability(
                record, agent_clues, board_words
            )
            clue_believabilities.append(believability)
        
        # Aggregate scores
        if not clue_believabilities:
            return {'composite_believability': 0.5}
        
        # Average each metric across all clues
        aggregated = {}
        metrics = clue_believabilities[0].keys()
        
        for metric in metrics:
            values = [b[metric] for b in clue_believabilities if metric in b]
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        return aggregated
    
    def get_comprehensive_rankings(self) -> EnhancedTournamentResults:
        """Generate comprehensive rankings with all metrics"""
        
        # Team rankings (sorted by conservative skill)
        team_rankings = []
        for team_key, stats in self.team_results.items():
            team_rankings.append((team_key, stats))
        team_rankings.sort(key=lambda x: x[1].conservative_skill, reverse=True)
        
        # Agent rankings (sorted by role-based TrueSkill)
        agent_rankings = []
        for agent_name, metrics in self.agent_metrics.items():
            agent_rankings.append((agent_name, metrics))
        agent_rankings.sort(key=lambda x: x[1].role_based_rating.mu, reverse=True)
        
        # Role-specific rankings
        codemaster_rankings = [(name, metrics) for name, metrics in agent_rankings
                              if metrics.agent_type == 'codemaster']
        guesser_rankings = [(name, metrics) for name, metrics in agent_rankings
                           if metrics.agent_type == 'guesser']
        
        # Believability analysis summary
        believability_analysis = self._generate_believability_summary()
        
        return EnhancedTournamentResults(
            tournament_name=self.tournament_name,
            total_games=len(self.match_results),
            match_results=self.match_results,
            team_rankings=team_rankings,
            agent_rankings=agent_rankings,
            codemaster_rankings=codemaster_rankings,
            guesser_rankings=guesser_rankings,
            believability_analysis=believability_analysis
        )
    
    def _generate_believability_summary(self) -> Dict[str, Any]:
        """Generate summary of believability analysis"""
        summary = {
            'total_clues_analyzed': len(self.all_clue_records),
            'agent_believability_scores': {},
            'top_believable_codemasters': [],
            'believability_distribution': {}
        }
        
        # Collect all composite scores
        all_scores = []
        for agent_name, metrics in self.agent_metrics.items():
            if metrics.agent_type == 'codemaster' and metrics.composite_believability > 0:
                score = metrics.composite_believability
                all_scores.append(score)
                summary['agent_believability_scores'][agent_name] = {
                    'composite_score': score,
                    'detailed_scores': metrics.believability_scores
                }
        
        # Top believable codemasters
        believable_cms = [(name, metrics.composite_believability) 
                         for name, metrics in self.agent_metrics.items()
                         if metrics.agent_type == 'codemaster']
        believable_cms.sort(key=lambda x: x[1], reverse=True)
        summary['top_believable_codemasters'] = believable_cms[:5]
        
        # Distribution statistics
        if all_scores:
            summary['believability_distribution'] = {
                'mean': np.mean(all_scores),
                'std': np.std(all_scores),
                'min': np.min(all_scores),
                'max': np.max(all_scores),
                'median': np.median(all_scores)
            }
        
        return summary
    
    def run_tournament(self, shuffle_matchups: bool = True):
        """Run the complete enhanced tournament"""
        print(f"Starting enhanced tournament: {self.tournament_name}")
        print(f"Registered codemasters: {[cm.name for cm in self.codemasters]}")
        print(f"Registered guessers: {[g.name for g in self.guessers]}")
        
        # Generate matchups
        matchups = self.generate_matchups()
        total_games = len(matchups) * self.games_per_matchup
        
        print(f"Total games to play: {total_games}")
        print(f"Estimated time: {total_games * 0.5 / 60:.1f} minutes")
        
        if shuffle_matchups:
            random.shuffle(matchups)
        
        game_count = 0
        
        # Run all games
        for red_team, blue_team in matchups:
            for game_num in range(self.games_per_matchup):
                game_count += 1
                print(f"\nGame {game_count}/{total_games}")
                
                # Generate unique seed
                game_seed = game_count * 1000 + random.randint(1, 999)
                
                # Run the game
                result = self.run_single_game(red_team, blue_team, game_seed)
                
                # Store result and update metrics
                if result.winner != "ERROR":
                    self.match_results.append(result)
                    self.update_comprehensive_metrics(result)
                
                # Save progress periodically
                if game_count % 10 == 0:
                    self.save_intermediate_results()
        
        # Calculate all final metrics
        print("\nCalculating final metrics...")
        self.calculate_all_trueskill_ratings()
        self.calculate_role_specific_metrics()
        self.analyze_believability()
        
        # Generate and save comprehensive results
        results = self.get_comprehensive_rankings()
        self.save_comprehensive_results(results)
        
        print(f"\nEnhanced tournament completed! Results saved to {self.results_dir}")
        return results
    
    def save_intermediate_results(self):
        """Save intermediate results during tournament"""
        results_file = os.path.join(self.results_dir, f"{self.tournament_name}_intermediate.json")
        
        data = {
            'match_results': [asdict(result) for result in self.match_results],
            'team_stats': {k: asdict(v) for k, v in self.team_results.items()},
            'agent_metrics': {k: asdict(v) for k, v in self.agent_metrics.items()},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_comprehensive_results(self, results: EnhancedTournamentResults):
        """Save comprehensive tournament results"""
        
        # Save detailed JSON results
        results_file = os.path.join(self.results_dir, f"{self.tournament_name}_comprehensive.json")
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        
        # Save human-readable rankings
        self._save_readable_rankings(results)
        
        # Save believability report
        self._save_believability_report(results)
        
        print(f"Comprehensive results saved to {results_file}")
    
    def _save_readable_rankings(self, results: EnhancedTournamentResults):
        """Save human-readable rankings report"""
        rankings_file = os.path.join(self.results_dir, f"{self.tournament_name}_rankings.txt")
        
        with open(rankings_file, 'w') as f:
            f.write(f"ENHANCED TOURNAMENT RANKINGS: {self.tournament_name}\n")
            f.write(f"Total Games: {results.total_games}\n")
            f.write("="*80 + "\n\n")
            
            # Team Rankings
            f.write("TEAM RANKINGS (by Conservative TrueSkill: μ - 3σ)\n")
            f.write("-" * 60 + "\n")
            for i, (team_key, stats) in enumerate(results.team_rankings[:10], 1):
                ci_low, ci_high = stats.wilson_confidence_interval
                f.write(f"{i:2d}. {team_key}\n")
                f.write(f"    Conservative Skill: {stats.conservative_skill:.2f}\n")
                f.write(f"    TrueSkill: {stats.trueskill_rating.mu:.2f} ± {stats.trueskill_rating.sigma:.2f}\n")
                f.write(f"    Win Rate: {stats.win_rate:.1%} ({stats.wins}-{stats.losses})\n")
                f.write(f"    Wilson 95% CI: [{ci_low:.3f}, {ci_high:.3f}]\n")
                f.write(f"    Avg Turns (Wins): {stats.avg_turns_when_winning:.1f}\n")
                f.write(f"    Avg Turns (All): {stats.avg_turns_all_games:.1f}\n")
                f.write(f"    Avg Words Found: {stats.avg_words_found:.1f}\n")
                f.write("\n")
            
            # Codemaster Rankings
            f.write("\nCODEMASTER RANKINGS (by Role-based TrueSkill)\n")
            f.write("-" * 60 + "\n")
            for i, (name, metrics) in enumerate(results.codemaster_rankings[:10], 1):
                f.write(f"{i:2d}. {name}\n")
                f.write(f"    Role TrueSkill: {metrics.role_based_rating.mu:.2f} ± {metrics.role_based_rating.sigma:.2f}\n")
                f.write(f"    Win Rate: {metrics.wins/max(1,metrics.games_played):.1%} ({metrics.wins}-{metrics.losses})\n")
                f.write(f"    Composite Believability: {metrics.composite_believability:.3f}\n")
                
                # Codemaster-specific metrics
                cm_metrics = metrics.codemaster_metrics
                if cm_metrics:
                    f.write(f"    Clue Efficiency: {cm_metrics.get('avg_clue_efficiency', 0):.3f}\n")
                    f.write(f"    One-step Solvability: {cm_metrics.get('one_step_solvability', 0):.1%}\n")
                    f.write(f"    Safety Rate: {cm_metrics.get('safety_rate', 0):.1%}\n")
                f.write("\n")
            
            # Guesser Rankings
            f.write("\nGUESSER RANKINGS (by Role-based TrueSkill)\n")
            f.write("-" * 60 + "\n")
            for i, (name, metrics) in enumerate(results.guesser_rankings[:10], 1):
                f.write(f"{i:2d}. {name}\n")
                f.write(f"    Role TrueSkill: {metrics.role_based_rating.mu:.2f} ± {metrics.role_based_rating.sigma:.2f}\n")
                f.write(f"    Win Rate: {metrics.wins/max(1,metrics.games_played):.1%} ({metrics.wins}-{metrics.losses})\n")
                
                # Guesser-specific metrics
                g_metrics = metrics.guesser_metrics
                if g_metrics:
                    f.write(f"    Guess Accuracy: {g_metrics.get('guess_accuracy', 0):.1%}\n")
                    f.write(f"    First Guess Accuracy: {g_metrics.get('first_guess_accuracy', 0):.1%}\n")
                    f.write(f"    Risk Management: {g_metrics.get('risk_management', 0):.1%}\n")
                f.write("\n")
    
    def _save_believability_report(self, results: EnhancedTournamentResults):
        """Save detailed believability analysis report"""
        report_file = os.path.join(self.results_dir, f"{self.tournament_name}_believability.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"BELIEVABILITY ANALYSIS REPORT: {self.tournament_name}\n")
            f.write("="*80 + "\n\n")
            
            analysis = results.believability_analysis
            
            f.write(f"Total Clues Analyzed: {analysis.get('total_clues_analyzed', 0)}\n\n")
            
            # Overall distribution
            dist = analysis.get('believability_distribution', {})
            if dist:
                f.write("BELIEVABILITY SCORE DISTRIBUTION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean: {dist.get('mean', 0):.3f}\n")
                f.write(f"Std Dev: {dist.get('std', 0):.3f}\n")
                f.write(f"Range: [{dist.get('min', 0):.3f}, {dist.get('max', 0):.3f}]\n")
                f.write(f"Median: {dist.get('median', 0):.3f}\n\n")
            
            # Top believable codemasters
            top_believable = analysis.get('top_believable_codemasters', [])
            if top_believable:
                f.write("TOP BELIEVABLE CODEMASTERS\n")
                f.write("-" * 40 + "\n")
                for i, (name, score) in enumerate(top_believable, 1):
                    f.write(f"{i}. {name}: {score:.3f}\n")
                f.write("\n")
            
            # Detailed agent scores
            agent_scores = analysis.get('agent_believability_scores', {})
            if agent_scores:
                f.write("DETAILED BELIEVABILITY SCORES\n")
                f.write("-" * 40 + "\n")
                for agent_name, scores in agent_scores.items():
                    f.write(f"\n{agent_name}:\n")
                    f.write(f"  Composite Score: {scores.get('composite_score', 0):.3f}\n")
                    
                    detailed = scores.get('detailed_scores', {})
                    if detailed:
                        f.write("  Component Scores:\n")
                        for metric, value in detailed.items():
                            if not metric.endswith('_zscore'):
                                f.write(f"    {metric}: {value:.3f}\n")

# Backwards compatibility
TournamentManager = EnhancedTournamentManager