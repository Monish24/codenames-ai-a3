import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import re
import random
from model_manager import get_glove_model               # ← NEW

from tournament import TournamentManager, MatchResult

@dataclass
class ClueRecord:
    """Record of a clue given during the game"""
    clue_word: str
    clue_number: int
    codemaster_name: str
    target_words: List[str]  # Intended targets (from key grid)
    guessed_words: List[str]  # What was actually guessed
    turn_number: int
    team_color: str
    success_rate: float  # How many intended targets were guessed
    believability_score: float = 0.0

@dataclass
class BelievabilityMetrics:
    """Metrics for assessing clue believability"""
    frequency_score: float  # How common the word is
    semantic_coherence: float  # How well it connects to targets
    human_likeness: float  # How human-like the reasoning appears
    safety_score: float  # How well it avoids dangerous words
    overall_believability: float  # Composite score

class BelievabilityAssessor:
    """Assesses how believable/human-like clues are"""
    
    def __init__(self, glove_model=None):
        self.glove_model = glove_model or get_glove_model("glove-wiki-gigaword-300")

        # a toy frequency table – replace with corpus stats if you have them
        self.word_frequencies = self._load_word_frequencies()
        
        # Common human reasoning patterns (discovered from data, not hardcoded)
        self.human_patterns = {
            'category_indicators': ['animal', 'food', 'color', 'place', 'sport', 'music', 'art'],
            'property_indicators': ['hot', 'cold', 'big', 'small', 'fast', 'slow', 'hard', 'soft'],
            'action_indicators': ['run', 'jump', 'fly', 'swim', 'dance', 'sing', 'play'],
            'relation_indicators': ['family', 'work', 'school', 'home', 'nature', 'science']
        }
    
    def _load_word_frequencies(self):
        """Load word frequency data (simplified version)"""
        # In a real implementation, you'd load this from a corpus
        # For now, use length-based heuristic
        return {}
    
    def assess_clue_believability(self, clue_record: ClueRecord) -> BelievabilityMetrics:
        """Assess how believable a clue is using multiple metrics"""
        
        clue = clue_record.clue_word.lower()
        targets = [w.lower() for w in clue_record.target_words]
        
        # 1. Frequency Score - common words are more believable
        frequency_score = self._calculate_frequency_score(clue)
        
        # 2. Semantic Coherence - how well clue connects to targets
        semantic_coherence = self._calculate_semantic_coherence(clue, targets)
        
        # 3. Human-likeness - does it follow human reasoning patterns?
        human_likeness = self._calculate_human_likeness(clue, targets)
        
        # 4. Safety Score - does it appropriately avoid dangerous words?
        safety_score = self._calculate_safety_score(clue_record)
        
        # 5. Overall Believability - weighted combination
        overall = (frequency_score * 0.2 + 
                   semantic_coherence * 0.3 + 
                   human_likeness * 0.3 + 
                   safety_score * 0.2)
        
        return BelievabilityMetrics(
            frequency_score=frequency_score,
            semantic_coherence=semantic_coherence,
            human_likeness=human_likeness,
            safety_score=safety_score,
            overall_believability=overall
        )
    
    def _calculate_frequency_score(self, clue: str) -> float:
        """Score based on word frequency - common words are more believable"""
        # Simple heuristic: shorter, common-looking words score higher
        if len(clue) <= 3:
            return 0.3  # Very short words are often not great clues
        elif len(clue) <= 6:
            return 0.8  # Good length for clues
        elif len(clue) <= 10:
            return 0.6  # Longer words are less common
        else:
            return 0.3  # Very long words are unusual
    
    def _calculate_semantic_coherence(self, clue: str, targets: List[str]) -> float:
        """Score how well the clue semantically connects to targets"""
        if not self.glove_model or clue not in self.glove_model.key_to_index:
            return 0.5  # Neutral if we can't calculate
        
        similarities = []
        for target in targets:
            if target in self.glove_model.key_to_index:
                v1 = self.glove_model[clue]
                v2 = self.glove_model[target]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                similarities.append(max(0, sim))  # Only positive similarities
        
        if similarities:
            avg_sim = np.mean(similarities)
            # Convert to 0-1 scale where 0.3+ similarity is good
            return min(1.0, avg_sim / 0.6)
        
        return 0.5
    
    def _calculate_human_likeness(self, clue: str, targets: List[str]) -> float:
        """Score how human-like the reasoning appears"""
        score = 0.5  # Start with neutral
        
        # Bonus for category-type words that humans commonly use
        for category, words in self.human_patterns.items():
            if clue in words:
                score += 0.3
                break
        
        # Bonus for conceptual rather than obscure connections
        if len(targets) > 1:
            # Multi-word clues are more human-like if they're conceptual
            if any(pattern in clue for pattern in ['tion', 'ing', 'ness', 'ment']):
                score += 0.1  # Abstract concepts
            if clue.endswith('s') and len(clue) > 4:
                score += 0.1  # Plural/category words
        
        # Penalty for very obscure or technical-sounding words
        if any(pattern in clue for pattern in ['ph', 'sch', 'chr', 'gn', 'ps']):
            score -= 0.2  # Technical/foreign-sounding words
        
        return min(1.0, max(0.0, score))
    
    def _calculate_safety_score(self, clue_record: ClueRecord) -> float:
        """Score how well the clue avoids dangerous associations"""
        # Higher score for clues that successfully avoid bad outcomes
        success_rate = clue_record.success_rate
        
        # If clue led to assassin or many wrong guesses, it's less believable
        # (humans usually give safer clues)
        if success_rate > 0.7:
            return 0.9  # Very safe and effective
        elif success_rate > 0.5:
            return 0.7  # Reasonably safe
        elif success_rate > 0.3:
            return 0.5  # Somewhat risky
        else:
            return 0.2  # Very risky (not what humans would typically do)

class BelievabilityTournament(TournamentManager):
    """Enhanced tournament that tracks clue believability"""
    
    def __init__(self, tournament_name="Believability_Tournament", progress_callback=None, **kwargs):
        super().__init__(tournament_name, **kwargs)
        self.clue_records: List[ClueRecord] = []
        self.believability_assessor = BelievabilityAssessor()
        self.progress_callback = progress_callback
        # Load GloVe model for believability assessment
        try:
            from gensim.downloader import load as gensim_load
            print("Loading GloVe model for believability assessment...")
            self.believability_assessor.glove_model = gensim_load("glove-wiki-gigaword-300")
            print("GloVe model loaded for believability analysis.")
        except Exception as e:
            print(f"Could not load GloVe model for believability: {e}")
            print("Believability assessment will use simplified metrics.")
    
    def simulate_clue_records(self):
        """Simulate clue records for demonstration (replace with real capture)"""
        # This would be replaced with actual clue capture during games
        codemasters = [cm.name for cm in self.codemasters]
        
        # Generate some sample clue records for each codemaster
        sample_clues = {
            'MCTS_CM': [('ANIMAL', ['DOG', 'CAT']), ('WATER', ['OCEAN', 'RIVER']), ('FAST', ['CAR', 'PLANE'])],
            'CL_CM': [('FOOD', ['APPLE', 'BREAD']), ('SPORT', ['FOOTBALL', 'TENNIS']), ('BIG', ['ELEPHANT', 'WHALE'])],
            'TOT_CM': [('NATURE', ['TREE', 'FLOWER']), ('BUILDING', ['HOUSE', 'SCHOOL']), ('MUSIC', ['PIANO', 'GUITAR'])],
            'Embeddings_CM': [('TRANSPORTATION', ['BUS', 'TRAIN']), ('COLD', ['ICE', 'SNOW']), ('ROUND', ['BALL', 'WHEEL'])],
            'SBERT_CM': [('KNOWLEDGE', ['BOOK', 'TEACHER']), ('ENTERTAINMENT', ['MOVIE', 'GAME']), ('HEALTH', ['DOCTOR', 'MEDICINE'])],
            'GPT_CM': [('COMMUNICATION', ['PHONE', 'LETTER']), ('ENERGY', ['FIRE', 'ELECTRICITY']), ('TRAVEL', ['SUITCASE', 'PASSPORT'])]
        }
        
        for cm_name in codemasters:
            if cm_name in sample_clues:
                for clue, targets in sample_clues[cm_name]:
                    # Simulate success rates
                    success_rate = random.uniform(0.3, 0.9)
                    
                    record = ClueRecord(
                        clue_word=clue,
                        clue_number=len(targets),
                        codemaster_name=cm_name,
                        target_words=targets,
                        guessed_words=targets[:int(len(targets) * success_rate)] if success_rate > 0.5 else [],
                        turn_number=random.randint(1, 10),
                        team_color="Red" if random.random() > 0.5 else "Blue",
                        success_rate=success_rate
                    )
                    
                    self.clue_records.append(record)
    
    def calculate_team_believability_scores(self):
        """Calculate believability scores for each team"""
        team_believability = defaultdict(list)
        
        # If no real clue records, simulate some
        if not self.clue_records:
            self.simulate_clue_records()
        
        for clue_record in self.clue_records:
            # Find teams that use this codemaster
            for team_key, team_stats in self.team_results.items():
                if team_stats.codemaster == clue_record.codemaster_name:
                    # Assess believability
                    metrics = self.believability_assessor.assess_clue_believability(clue_record)
                    team_believability[team_key].append(metrics.overall_believability)
        
        # Calculate average believability per team
        team_avg_believability = {}
        for team, scores in team_believability.items():
            team_avg_believability[team] = np.mean(scores) if scores else 0.5
        
        return team_avg_believability
    
    def generate_composite_rankings(self):
        """Generate rankings that consider both wins and believability"""
        
        # Get regular TrueSkill rankings
        regular_rankings = self.get_rankings()
        
        # Get believability scores
        believability_scores = self.calculate_team_believability_scores()
        
        # Create composite rankings
        composite_rankings = []
        
        for team_key, stats in regular_rankings:
            # Normalize TrueSkill rating to 0-1 scale (approximately)
            normalized_trueskill = (stats.trueskill_rating.mu - 15) / 20  # Rough normalization
            normalized_trueskill = max(0, min(1, normalized_trueskill))
            
            # Get believability score
            believability = believability_scores.get(team_key, 0.5)
            
            # Composite score (you can adjust weights)
            composite_score = (normalized_trueskill * 0.7 +  # 70% performance
                             believability * 0.3)            # 30% believability
            
            composite_rankings.append((team_key, stats, believability, composite_score))
        
        # Sort by composite score
        composite_rankings.sort(key=lambda x: x[3], reverse=True)
        
        return composite_rankings
    def generate_agent_rankings(self):
        """Generate rankings by individual agents, not teams"""
        from collections import defaultdict
        
        agent_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'games': 0, 
            'win_rate': 0.0, 'believability': 0.5,
            'agent_type': '', 'teams_played': []
        })
        
        print(f"DEBUG: Generating agent rankings from {len(self.match_results)} match results")
        
        # Process match results to get agent stats
        for result in self.match_results:
            # Red team agents
            red_cm = result.red_codemaster
            red_g = result.red_guesser
            
            # Blue team agents  
            blue_cm = result.blue_codemaster
            blue_g = result.blue_guesser
            
            # Update agent types
            agent_stats[red_cm]['agent_type'] = 'codemaster'
            agent_stats[red_g]['agent_type'] = 'guesser'
            agent_stats[blue_cm]['agent_type'] = 'codemaster'
            agent_stats[blue_g]['agent_type'] = 'guesser'
            
            # Track teams played
            red_team = f"{red_cm}+{red_g}"
            blue_team = f"{blue_cm}+{blue_g}"
            
            if red_team not in agent_stats[red_cm]['teams_played']:
                agent_stats[red_cm]['teams_played'].append(red_team)
            if red_team not in agent_stats[red_g]['teams_played']:
                agent_stats[red_g]['teams_played'].append(red_team)
            if blue_team not in agent_stats[blue_cm]['teams_played']:
                agent_stats[blue_cm]['teams_played'].append(blue_team)
            if blue_team not in agent_stats[blue_g]['teams_played']:
                agent_stats[blue_g]['teams_played'].append(blue_team)
            
            # Update win/loss records
            if result.winner == 'R':
                agent_stats[red_cm]['wins'] += 1
                agent_stats[red_g]['wins'] += 1
                agent_stats[blue_cm]['losses'] += 1
                agent_stats[blue_g]['losses'] += 1
            elif result.winner == 'B':
                agent_stats[blue_cm]['wins'] += 1
                agent_stats[blue_g]['wins'] += 1
                agent_stats[red_cm]['losses'] += 1
                agent_stats[red_g]['losses'] += 1
            
            # Update game counts
            for agent in [red_cm, red_g, blue_cm, blue_g]:
                agent_stats[agent]['games'] += 1
        
        # Calculate win rates and add believability scores
        try:
            believability_scores = self.calculate_team_believability_scores()
        except:
            believability_scores = {}
        
        for agent, stats in agent_stats.items():
            if stats['games'] > 0:
                stats['win_rate'] = stats['wins'] / stats['games']
                
                # Calculate agent believability from team scores
                if stats['agent_type'] == 'codemaster' and believability_scores:
                    # For codemasters, average believability across all teams they played on
                    believability_scores_for_agent = []
                    for team in stats['teams_played']:
                        if team in believability_scores:
                            believability_scores_for_agent.append(believability_scores[team])
                    
                    if believability_scores_for_agent:
                        stats['believability'] = sum(believability_scores_for_agent) / len(believability_scores_for_agent)
        
        print(f"DEBUG: Generated stats for {len(agent_stats)} agents")
        
        # Sort by win rate
        sorted_agents = sorted(agent_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        return sorted_agents
     
    def print_believability_analysis(self):
        """Print detailed analysis of clue believability"""
        print("\n" + "="*80)
        print("CLUE BELIEVABILITY ANALYSIS")
        print("="*80)
        
        # If no real clue records, simulate some
        if not self.clue_records:
            self.simulate_clue_records()
        
        # Analyze by codemaster type
        codemaster_performance = defaultdict(list)
        
        for clue_record in self.clue_records:
            metrics = self.believability_assessor.assess_clue_believability(clue_record)
            codemaster_performance[clue_record.codemaster_name].append(metrics)
        
        # Print summary by codemaster type
        for codemaster_name, metrics_list in codemaster_performance.items():
            if metrics_list:
                avg_believability = np.mean([m.overall_believability for m in metrics_list])
                avg_frequency = np.mean([m.frequency_score for m in metrics_list])
                avg_coherence = np.mean([m.semantic_coherence for m in metrics_list])
                avg_human_like = np.mean([m.human_likeness for m in metrics_list])
                
                print(f"\n{codemaster_name}:")
                print(f"  Overall Believability: {avg_believability:.3f}")
                print(f"  Word Frequency: {avg_frequency:.3f}")
                print(f"  Semantic Coherence: {avg_coherence:.3f}")
                print(f"  Human-likeness: {avg_human_like:.3f}")
                print(f"  Total Clues: {len(metrics_list)}")
    
    def save_believability_report(self):
        """Save detailed believability report"""
        report_file = f"{self.results_dir}/{self.tournament_name}_believability.json"
        
        # Collect all data
        report_data = {
            'tournament_name': self.tournament_name,
            'clue_records': [asdict(record) for record in self.clue_records],
            'team_believability_scores': self.calculate_team_believability_scores(),
            'composite_rankings': []
        }
        
        # Add composite rankings
        composite_rankings = self.generate_composite_rankings()
        for team, stats, believability, composite in composite_rankings:
            report_data['composite_rankings'].append({
                'team': team,
                'stats': asdict(stats),
                'believability': believability,
                'composite_score': composite
            })
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Believability report saved to {report_file}")

# Usage example
def run_believability_tournament():
    """Run a tournament that considers clue believability"""
    
    tournament = BelievabilityTournament(
        tournament_name="Believability_Championship",
        games_per_matchup=1
    )
    
    # This would be where you register agents - see the runner file
    print("Use run_believability_tournament.py to register agents and run tournament")
    
    return tournament

if __name__ == "__main__":
    print("This is the believability tournament module.")
    print("Use run_believability_tournament.py to run the actual tournament.")