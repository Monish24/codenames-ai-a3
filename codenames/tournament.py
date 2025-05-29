import json
import random
import time
import os
import itertools
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import trueskill
import numpy as np

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
    
    def __post_init__(self):
        if self.trueskill_rating is None:
            self.trueskill_rating = trueskill.Rating()

class TournamentManager:
    
    def __init__(self, 
                 tournament_name: str = "Codenames_Tournament",
                 results_dir: str = "tournament_results",
                 games_per_matchup: int = 1,
                 max_matchups: int = 500):  

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
        
        # TrueSkill environment
        self.ts_env = trueskill.TrueSkill()
    
    def register_agent(self, name: str, agent_type: str, class_reference: Any, **kwargs):
        agent = Agent(name, agent_type, class_reference, kwargs)
        
        if agent_type == 'codemaster':
            self.codemasters.append(agent)
        elif agent_type == 'guesser':
            self.guessers.append(agent)
        else:
            raise ValueError(f"Invalid agent_type: {agent_type}")
        
        print(f"Registered {agent_type}: {name}")
    
    def generate_team_combinations(self) -> List[Tuple[Agent, Agent]]:
        teams = []
        for cm in self.codemasters:
            for g in self.guessers:
                teams.append((cm, g))
        return teams
    
    def generate_matchups(self) -> List[Tuple[Tuple[Agent, Agent], Tuple[Agent, Agent]]]:
        teams = self.generate_team_combinations()
        all_matchups = []
        
        print(f"Total teams: {len(teams)}")
        print(f"Max theoretical matchups: {len(teams) * (len(teams) - 1)}")
        
        # Generate all possible matchups first
        for i, team_a in enumerate(teams):
            for j, team_b in enumerate(teams):
                if i != j:  # Teams don't play against themselves
                    all_matchups.append((team_a, team_b))
        
        print(f"Total possible matchups: {len(all_matchups)}")
        
        # If we have too many matchups, intelligently reduce them
        if len(all_matchups) > self.max_matchups:
            print(f"Reducing from {len(all_matchups)} to {self.max_matchups} matchups...")
            
            # Strategy: Ensure each team plays a minimum number of games
            final_matchups = []
            team_game_count = defaultdict(int)
            
            # Shuffle for randomness
            random.shuffle(all_matchups)
            
            # First pass: ensure each team plays at least 3 games
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
            
            # Second pass: fill remaining slots randomly
            remaining_matchups = [m for m in all_matchups if m not in final_matchups]
            random.shuffle(remaining_matchups)
            
            slots_remaining = self.max_matchups - len(final_matchups)
            final_matchups.extend(remaining_matchups[:slots_remaining])
            
            print(f"Final matchups selected: {len(final_matchups)}")
            print("Team game distribution:")
            for team, count in sorted(team_game_count.items()):
                print(f"  {team}: {count} games")
            
            return final_matchups
        else:
            print(f"Using all {len(all_matchups)} matchups")
            return all_matchups
    
    def run_single_game(self, 
                    red_team: Tuple[Agent, Agent], 
                    blue_team: Tuple[Agent, Agent],
                    game_seed: int = None) -> MatchResult:
        """Run a single game between two teams"""
        
        red_cm, red_g = red_team
        blue_cm, blue_g = blue_team
        
        if game_seed is None:
            game_seed = random.randint(1, 1000000)
        
        # Create match ID
        match_id = f"{red_cm.name}+{red_g.name}_vs_{blue_cm.name}+{blue_g.name}_{game_seed}"
        
        print(f"Running match: {match_id}")
        
        start_time = time.time()
        
        try:
            # Create game instance
            game = Game(
                codemaster_red=red_cm.class_reference,
                guesser_red=red_g.class_reference,
                codemaster_blue=blue_cm.class_reference,
                guesser_blue=blue_g.class_reference,
                seed=game_seed,
                do_print=True,  # Suppress game output during tournament
                do_log=False,    # We'll handle logging ourselves
                cmr_kwargs=red_cm.kwargs,
                gr_kwargs=red_g.kwargs,
                cmb_kwargs=blue_cm.kwargs,
                gb_kwargs=blue_g.kwargs
            )
            
            # Run the game
            game.run()
            
            end_time = time.time()
            
            # Extract results
            red_words = game.words_on_board.count("*Red*")
            blue_words = game.words_on_board.count("*Blue*")
            civilians = game.words_on_board.count("*Civilian*")
            assassin = game.words_on_board.count("*Assassin*") > 0
            
            # Create result object
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
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            return result
            
        except Exception as e:
            print(f"Error in match {match_id}: {str(e)}")
            # Return a default result indicating an error
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
    
    def run_tournament(self, shuffle_matchups: bool = True):
        """Run the complete tournament with reasonable limits"""
        print(f"Starting tournament: {self.tournament_name}")
        print(f"Registered codemasters: {[cm.name for cm in self.codemasters]}")
        print(f"Registered guessers: {[g.name for g in self.guessers]}")
        print(f"Maximum matchups allowed: {self.max_matchups}")
        
        # Generate limited matchups
        matchups = self.generate_matchups()
        total_games = len(matchups) * self.games_per_matchup
        
        print(f"Final matchups: {len(matchups)}")
        print(f"Games per matchup: {self.games_per_matchup}")
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
                
                # Generate unique seed for each game
                game_seed = game_count * 1000 + random.randint(1, 999)
                
                # Run the game
                result = self.run_single_game(red_team, blue_team, game_seed)
                
                # Store result
                if result.winner != "ERROR":
                    self.match_results.append(result)
                    self.update_team_stats(result)
                
                # Save progress periodically
                if game_count % 10 == 0:
                    self.save_intermediate_results()
        
        # Calculate final rankings
        self.calculate_trueskill_ratings()
        
        # Save final results
        self.save_final_results()
        
        print(f"\nTournament completed! Results saved to {self.results_dir}")
    
    def update_team_stats(self, result: MatchResult):
        """Update team statistics based on game result"""
        red_team_key = f"{result.red_codemaster}+{result.red_guesser}"
        blue_team_key = f"{result.blue_codemaster}+{result.blue_guesser}"
        
        # Initialize team results if not exists
        if red_team_key not in self.team_results:
            self.team_results[red_team_key] = TeamResult(
                codemaster=result.red_codemaster,
                guesser=result.red_guesser
            )
        
        if blue_team_key not in self.team_results:
            self.team_results[blue_team_key] = TeamResult(
                codemaster=result.blue_codemaster,
                guesser=result.blue_guesser
            )
        
        red_stats = self.team_results[red_team_key]
        blue_stats = self.team_results[blue_team_key]
        
        # Update win/loss records
        if result.winner == 'R':
            red_stats.wins += 1
            blue_stats.losses += 1
        elif result.winner == 'B':
            blue_stats.wins += 1
            red_stats.losses += 1
        
        # Update game counts
        red_stats.total_games += 1
        blue_stats.total_games += 1
        
        # Update averages
        red_stats.avg_words_found = (red_stats.avg_words_found * (red_stats.total_games - 1) + 
                                    result.red_words_found) / red_stats.total_games
        blue_stats.avg_words_found = (blue_stats.avg_words_found * (blue_stats.total_games - 1) + 
                                     result.blue_words_found) / blue_stats.total_games
    
    def calculate_trueskill_ratings(self):
        """Calculate TrueSkill ratings for all teams"""
        print("Calculating TrueSkill ratings...")
        
        # Initialize ratings
        for team_key in self.team_results:
            self.team_results[team_key].trueskill_rating = trueskill.Rating()
        
        # Process each match result
        for result in self.match_results:
            red_team_key = f"{result.red_codemaster}+{result.red_guesser}"
            blue_team_key = f"{result.blue_codemaster}+{result.blue_guesser}"
            
            red_rating = self.team_results[red_team_key].trueskill_rating
            blue_rating = self.team_results[blue_team_key].trueskill_rating
            
            # Determine ranks (0 = winner, 1 = loser)
            if result.winner == 'R':
                ranks = [0, 1]  # Red wins
            elif result.winner == 'B':
                ranks = [1, 0]  # Blue wins
            else:
                continue  # Skip draws/errors
            
            # Update ratings
            new_red, new_blue = trueskill.rate([(red_rating,), (blue_rating,)], ranks=ranks)
            
            self.team_results[red_team_key].trueskill_rating = new_red[0]
            self.team_results[blue_team_key].trueskill_rating = new_blue[0]
    
    def get_rankings(self) -> List[Tuple[str, TeamResult]]:
        """Get final rankings sorted by TrueSkill rating"""
        rankings = []
        for team_key, stats in self.team_results.items():
            rankings.append((team_key, stats))
        
        # Sort by TrueSkill rating (higher is better)
        rankings.sort(key=lambda x: x[1].trueskill_rating.mu, reverse=True)
        return rankings
    
    def save_intermediate_results(self):
        """Save intermediate results during tournament"""
        results_file = os.path.join(self.results_dir, f"{self.tournament_name}_intermediate.json")
        
        data = {
            'match_results': [asdict(result) for result in self.match_results],
            'team_stats': {k: asdict(v) for k, v in self.team_results.items()},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_final_results(self):
        """Save final tournament results"""
        # Save detailed results
        results_file = os.path.join(self.results_dir, f"{self.tournament_name}_final.json")
        
        data = {
            'tournament_name': self.tournament_name,
            'total_games': len(self.match_results),
            'games_per_matchup': self.games_per_matchup,
            'max_matchups': self.max_matchups,
            'match_results': [asdict(result) for result in self.match_results],
            'team_stats': {k: asdict(v) for k, v in self.team_results.items()},
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save rankings
        rankings = self.get_rankings()
        rankings_file = os.path.join(self.results_dir, f"{self.tournament_name}_rankings.txt")
        
        with open(rankings_file, 'w') as f:
            f.write(f"TOURNAMENT RANKINGS: {self.tournament_name}\n")
            f.write(f"Max matchups: {self.max_matchups}, Games per matchup: {self.games_per_matchup}\n")
            f.write("="*60 + "\n\n")
            
            for i, (team_key, stats) in enumerate(rankings, 1):
                win_rate = stats.wins / max(1, stats.total_games) * 100
                f.write(f"{i:2d}. {team_key}\n")
                f.write(f"    TrueSkill: {stats.trueskill_rating.mu:.2f} ± {stats.trueskill_rating.sigma:.2f}\n")
                f.write(f"    Record: {stats.wins}-{stats.losses} ({win_rate:.1f}%)\n")
                f.write(f"    Games Played: {stats.total_games}\n")
                f.write(f"    Avg Words Found: {stats.avg_words_found:.1f}\n")
                f.write("\n")
        
        print(f"Final results saved to {results_file}")
        print(f"Rankings saved to {rankings_file}")