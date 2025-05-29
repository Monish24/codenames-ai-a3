#!/usr/bin/env python3
"""
Enhanced Tournament Runner for Codenames AI
Supports both basic performance tournaments and believability analysis tournaments
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import tournament systems
from tournament import EnhancedTournamentManager
from believability_tournament import EnhancedBelievabilityTournament

def import_agent_safely(module_path: str, class_name: str):
    """Safely import an agent class with error handling"""
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[class_name])
        agent_class = getattr(module, class_name)
        return agent_class
    except ImportError as e:
        print(f"ImportError for {module_path}.{class_name}: {e}")
        return None
    except AttributeError as e:
        print(f"AttributeError for {module_path}.{class_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error importing {module_path}.{class_name}: {e}")
        return None

def load_available_agents() -> Dict[str, Dict[str, Any]]:
    """Load all available agents with their import information"""
    
    agents = {
        'codemasters': {
            'MCTS': {
                'module': 'players.codemasterMCTS',
                'class': 'CodemasterMCTS',
                'description': 'Monte Carlo Tree Search Codemaster',
                'kwargs': {'num_simulations': 100}
            },
            'EMD': {
                'module': 'players.codemaster_EMD',
                'class': 'CodemasterEmbeddings',
                'description': 'Word Embeddings Codemaster (Original)',
                'kwargs': {}
            },
            'SBERT': {
                'module': 'players.codemaster_SBERT',
                'class': 'CodemasterSBERT',
                'description': 'Sentence Transformers Codemaster',
                'kwargs': {}
            },
            'CL': {
                'module': 'players.codemaster_CL',
                'class': 'CodemasterCurriculum',
                'description': 'Curriculum Learning Codemaster',
                'kwargs': {}
            },
            'TOT': {
                'module': 'players.codemaster_TOT',
                'class': 'CodemasterTreeOfThoughts',
                'description': 'Tree of Thoughts Codemaster',
                'kwargs': {}
            }
        },
        'guessers': {
            'EMD': {
                'module': 'players.guesserEMD',
                'class': 'GuesserEmbeddings',
                'description': 'Word Embeddings Guesser',
                'kwargs': {}
            },
            'MCTS': {
                'module': 'players.guesser_MCTS',
                'class': 'GuesserMCTS',
                'description': 'Monte Carlo Tree Search Guesser',
                'kwargs': {'num_simulations': 50}
            },
            'SBERT': {
                'module': 'players.guesser_SBERT',
                'class': 'GuesserSBERT',
                'description': 'Sentence Transformers Guesser',
                'kwargs': {}
            },
            'Naive': {
                'module': 'players.guesser_naive',
                'class': 'NaiveGuesser',
                'description': 'Simple Embedding Guesser',
                'kwargs': {}
            }
        }
    }
    
    return agents

def test_agent_imports(agents: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Test which agents can be imported successfully"""
    
    available = {'codemasters': [], 'guessers': []}
    
    print("Testing agent imports...")
    print("-" * 40)
    
    for role in ['codemasters', 'guessers']:
        print(f"\n{role.upper()}:")
        for agent_code, agent_info in agents[role].items():
            agent_class = import_agent_safely(agent_info['module'], agent_info['class'])
            if agent_class is not None:
                try:
                    # Try to instantiate
                    instance = agent_class()
                    available[role].append(agent_code)
                    print(f"  ✅ {agent_code}: {agent_info['description']}")
                except Exception as e:
                    print(f"  ❌ {agent_code}: Import OK but instantiation failed - {str(e)}")
            else:
                print(f"  ❌ {agent_code}: Import failed")
    
    print(f"\nSummary: {len(available['codemasters'])} codemasters, {len(available['guessers'])} guessers available")
    return available

def register_agents_to_tournament(tournament, agents: Dict[str, Dict[str, Any]], 
                                selected_agents: Dict[str, List[str]]):
    """Register selected agents to the tournament"""
    
    registered_count = 0
    
    # Register codemasters
    for agent_code in selected_agents['codemasters']:
        if agent_code in agents['codemasters']:
            agent_info = agents['codemasters'][agent_code]
            agent_class = import_agent_safely(agent_info['module'], agent_info['class'])
            
            if agent_class:
                tournament.register_agent(
                    name=f"{agent_code}_CM",
                    agent_type='codemaster',
                    class_reference=agent_class,
                    **agent_info['kwargs']
                )
                registered_count += 1
    
    # Register guessers
    for agent_code in selected_agents['guessers']:
        if agent_code in agents['guessers']:
            agent_info = agents['guessers'][agent_code]
            agent_class = import_agent_safely(agent_info['module'], agent_info['class'])
            
            if agent_class:
                tournament.register_agent(
                    name=f"{agent_code}_Guesser",
                    agent_type='guesser',
                    class_reference=agent_class,
                    **agent_info['kwargs']
                )
                registered_count += 1
    
    return registered_count

def run_performance_tournament(args):
    """Run a performance-focused tournament"""
    
    print("=" * 60)
    print("CODENAMES AI PERFORMANCE TOURNAMENT")
    print("=" * 60)
    
    agents = load_available_agents()
    available = test_agent_imports(agents)
    
    if len(available['codemasters']) < 2 or len(available['guessers']) < 1:
        print(f"\n❌ Insufficient agents available!")
        print(f"Need at least 2 codemasters and 1 guesser.")
        return False
    
    # Create tournament
    tournament = EnhancedTournamentManager(
        tournament_name=args.name,
        games_per_matchup=args.games_per_matchup,
        max_matchups=args.max_matchups
    )
    
    # Select agents based on arguments
    if args.agents == 'all':
        selected_agents = available
    elif args.agents == 'core':
        selected_agents = {
            'codemasters': [a for a in ['MCTS', 'EMD', 'CL', 'TOT'] if a in available['codemasters']],
            'guessers': [a for a in ['EMD', 'MCTS'] if a in available['guessers']]
        }
    else:
        # Custom selection
        selected_agents = available  # Default to all for now
    
    # Register agents
    registered_count = register_agents_to_tournament(tournament, agents, selected_agents)
    
    if registered_count < 3:
        print(f"❌ Only {registered_count} agents registered successfully. Need at least 3.")
        return False
    
    # Calculate tournament size
    num_teams = len(tournament.codemasters) * len(tournament.guessers)
    total_matchups = min(num_teams * (num_teams - 1), args.max_matchups)
    total_games = total_matchups * args.games_per_matchup
    
    print(f"\n🏆 TOURNAMENT CONFIGURATION")
    print(f"Tournament Name: {args.name}")
    print(f"Registered Agents: {registered_count}")
    print(f"Teams: {num_teams}")
    print(f"Matchups: {total_matchups}")
    print(f"Total Games: {total_games}")
    print(f"Estimated Time: {total_games * 0.5 / 60:.1f} minutes")
    
    if not args.auto_confirm:
        confirm = input(f"\nProceed with tournament? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Tournament cancelled.")
            return False
    
    # Run tournament
    print(f"\n🚀 Starting tournament...")
    start_time = time.time()
    
    try:
        results = tournament.run_tournament(shuffle_matchups=True)
        end_time = time.time()
        
        print(f"\n✅ Tournament completed in {(end_time - start_time) / 60:.1f} minutes!")
        
        # Print top results
        print("\n🏆 TOP TEAM RANKINGS:")
        for i, (team_key, stats) in enumerate(results.team_rankings[:5], 1):
            ci_low, ci_high = stats.wilson_confidence_interval
            conservative_skill = stats.conservative_skill
            print(f"{i}. {team_key}")
            print(f"   Conservative Skill: {conservative_skill:.2f}")
            print(f"   Win Rate: {stats.win_rate:.1%} ({stats.wins}-{stats.losses})")
            print(f"   Wilson 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        
        print(f"\n📁 Results saved to: {tournament.results_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Tournament error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_believability_tournament(args):
    """Run a believability-focused tournament"""
    
    print("=" * 60)
    print("CODENAMES AI BELIEVABILITY TOURNAMENT")
    print("=" * 60)
    
    agents = load_available_agents()
    available = test_agent_imports(agents)
    
    if len(available['codemasters']) < 2 or len(available['guessers']) < 1:
        print(f"\n❌ Insufficient agents available!")
        return False
    
    # Create believability tournament
    tournament = EnhancedBelievabilityTournament(
        tournament_name=args.name,
        games_per_matchup=args.games_per_matchup,
        max_matchups=args.max_matchups
    )
    
    # Select agents (focus on codemasters for believability)
    if args.agents == 'all':
        selected_agents = available
    elif args.agents == 'core':
        selected_agents = {
            'codemasters': [a for a in ['MCTS', 'EMD', 'CL', 'TOT'] if a in available['codemasters']],
            'guessers': [a for a in ['EMD', 'Naive'] if a in available['guessers']]  # Simpler guessers
        }
    else:
        selected_agents = available
    
    # Register agents
    registered_count = register_agents_to_tournament(tournament, agents, selected_agents)
    
    if registered_count < 3:
        print(f"❌ Only {registered_count} agents registered successfully.")
        return False
    
    # Tournament info
    num_teams = len(tournament.codemasters) * len(tournament.guessers)
    total_matchups = min(num_teams * (num_teams - 1), args.max_matchups)
    total_games = total_matchups * args.games_per_matchup
    
    print(f"\n🧠 BELIEVABILITY TOURNAMENT CONFIGURATION")
    print(f"Tournament Name: {args.name}")
    print(f"Codemasters: {len(tournament.codemasters)} (believability analysis focus)")
    print(f"Guessers: {len(tournament.guessers)}")
    print(f"Teams: {num_teams}")
    print(f"Total Games: {total_games}")
    print(f"Estimated Time: {total_games * 0.6 / 60:.1f} minutes (includes analysis)")
    
    if not args.auto_confirm:
        confirm = input(f"\nProceed with believability tournament? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Tournament cancelled.")
            return False
    
    # Run tournament with believability analysis
    print(f"\n🚀 Starting believability tournament...")
    start_time = time.time()
    
    try:
        results = tournament.run_tournament_with_believability(shuffle_matchups=True)
        end_time = time.time()
        
        print(f"\n✅ Believability tournament completed in {(end_time - start_time) / 60:.1f} minutes!")
        
        # Print top believable codemasters
        print("\n🧠 TOP BELIEVABLE CODEMASTERS:")
        believability_analysis = results.believability_analysis
        top_believable = believability_analysis.get('top_believable_codemasters', [])
        
        for i, (name, score) in enumerate(top_believable[:5], 1):
            print(f"{i}. {name}: {score:.3f} believability")
        
        # Performance vs Believability
        perf_believe = believability_analysis.get('performance_believability_analysis', {})
        if 'overall_correlation' in perf_believe:
            overall_corr = perf_believe['overall_correlation']['mean']
            print(f"\n📊 Performance-Believability Correlation: {overall_corr:.3f}")
        
        print(f"\n📁 Enhanced results saved to: {tournament.results_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Tournament error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description="Enhanced Codenames AI Tournament Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Tournament type
    parser.add_argument(
        'tournament_type',
        choices=['performance', 'believability', 'test'],
        help='Type of tournament to run'
    )
    
    # Tournament configuration
    parser.add_argument(
        '--name',
        default=f"Tournament_{int(time.time())}",
        help='Tournament name'
    )
    
    parser.add_argument(
        '--games-per-matchup',
        type=int,
        default=2,
        help='Number of games each pair of teams plays'
    )
    
    parser.add_argument(
        '--max-matchups',
        type=int,
        default=300,
        help='Maximum number of matchups to run'
    )
    
    # Agent selection
    parser.add_argument(
        '--agents',
        choices=['all', 'core', 'custom'],
        default='core',
        help='Which agents to include'
    )
    
    # Execution options
    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Set tournament name based on type if not specified
    if args.name == f"Tournament_{int(time.time())}":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.name = f"{args.tournament_type.capitalize()}_Tournament_{timestamp}"
    
    print(f"Enhanced Codenames AI Tournament Runner")
    print(f"Tournament Type: {args.tournament_type}")
    print(f"Configuration: {args.games_per_matchup} games/matchup, max {args.max_matchup} matchups")
    
    # Run appropriate tournament type
    if args.tournament_type == 'performance':
        success = run_performance_tournament(args)
    elif args.tournament_type == 'believability':
        success = run_believability_tournament(args)
    elif args.tournament_type == 'test':
        # Just test imports
        agents = load_available_agents()
        available = test_agent_imports(agents)
        success = len(available['codemasters']) >= 2 and len(available['guessers']) >= 1
        if success:
            print("\n✅ All agent imports successful! Ready to run tournaments.")
        else:
            print("\n❌ Some agent imports failed. Fix errors before running tournaments.")
    else:
        print(f"❌ Unknown tournament type: {args.tournament_type}")
        success = False
    
    if success:
        print("\n🎉 Tournament runner completed successfully!")
        return 0
    else:
        print("\n❌ Tournament runner failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())