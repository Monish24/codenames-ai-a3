import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the believability tournament
from believability_tournament import BelievabilityTournament

# Import all your codemaster agents
try:
    from players.codemasterMCTS import CodemasterMCTS
    print("CodemasterMCTS imported")
except ImportError as e:
    print(f"CodemasterMCTS: {e}")
    CodemasterMCTS = None

try:
    from players.codemaster_GPT import CodemasterGPT
    print("CodemasterGPT imported")
except ImportError as e:
    print(f" CodemasterGPT: {e}")
    CodemasterGPT = None

try:
    from players.codemaster_EMD import CodemasterEmbeddings
    print(" CodemasterEmbeddings imported")
except ImportError as e:
    print(f" CodemasterEmbeddings: {e}")
    CodemasterEmbeddings = None

try:
    from players.codemaster_SBERT import CodemasterSBERT
    print(" CodemasterSBERT imported")
except ImportError as e:
    print(f" CodemasterSBERT: {e}")
    CodemasterSBERT = None

try:
    from players.codemaster_CL import CodemasterCurriculum
    print(" CodemasterCurriculum imported")
except ImportError as e:
    print(f" CodemasterCurriculum: {e}")
    CodemasterCurriculum = None

try:
    from players.codemaster_TOT import CodemasterTreeOfThoughts
    print(" CodemasterTreeOfThoughts imported")
except ImportError as e:
    print(f" CodemasterTreeOfThoughts: {e}")
    CodemasterTreeOfThoughts = None

# Import all your guesser agents
try:
    from players.guesserEMD import GuesserEmbeddings
    print(" GuesserEmbeddings imported")
except ImportError as e:
    print(f" GuesserEmbeddings: {e}")
    GuesserEmbeddings = None

try:
    from players.guesser_naive import NaiveGuesser
    print(" NaiveGuesser imported")
except ImportError as e:
    print(f" NaiveGuesser: {e}")
    NaiveGuesser = None

try:
    from players.guesser_SBERT import GuesserSBERT
    print(" GuesserSBERT imported")
except ImportError as e:
    print(f" GuesserSBERT: {e}")
    GuesserSBERT = None

try:
    from players.guesser_GPT import GuesserGPT
    print(" GuesserGPT imported")
except ImportError as e:
    print(f" GuesserGPT: {e}")
    GuesserGPT = None

try:
    from players.guesser_MCTS import GuesserMCTS
    print(" GuesserMCTS imported")
except ImportError as e:
    print(f" GuesserMCTS: {e}")
    GuesserMCTS = None

def count_available_agents():
    """Count how many agents are available"""
    codemasters = [CodemasterMCTS, CodemasterGPT, CodemasterEmbeddings, 
                  CodemasterSBERT, CodemasterCurriculum, CodemasterTreeOfThoughts]
    guessers = [GuesserEmbeddings, NaiveGuesser, GuesserSBERT, GuesserGPT, GuesserMCTS]
    
    available_cm = sum(1 for cm in codemasters if cm is not None)
    available_g = sum(1 for g in guessers if g is not None)
    
    return available_cm, available_g

def run_complete_believability_tournament():
    """Run comprehensive tournament with believability tracking"""
    
    print("Setting up Complete Believability Tournament...")
    
    # Check available agents
    available_cm, available_g = count_available_agents()
    
    if available_cm < 2 or available_g < 2:
        print(f" Need at least 2 codemasters and 2 guessers. Found {available_cm} CM, {available_g} G")
        print("Please fix import errors above.")
        return
    
    # Create enhanced tournament
    tournament = BelievabilityTournament(
        tournament_name="Complete_Believability_Championship",
        games_per_matchup=2  # Start with 2 games per matchup
    )
    
    print("\nRegistering Codemasters...")
    
    # Register all available codemasters
    if CodemasterMCTS:
        tournament.register_agent("MCTS_CM", "codemaster", CodemasterMCTS, 
                                 num_simulations=100)
    
    if CodemasterGPT:
        tournament.register_agent("GPT_CM", "codemaster", CodemasterGPT)
    
    if CodemasterEmbeddings:
        tournament.register_agent("Embeddings_CM", "codemaster", CodemasterEmbeddings)
    
    if CodemasterSBERT:
        tournament.register_agent("SBERT_CM", "codemaster", CodemasterSBERT)
    
    if CodemasterCurriculum:
        tournament.register_agent("CL_CM", "codemaster", CodemasterCurriculum)
    
    if CodemasterTreeOfThoughts:
        tournament.register_agent("TOT_CM", "codemaster", CodemasterTreeOfThoughts)
    
    print("Registering Guessers...")
    
    # Register all available guessers
    if GuesserEmbeddings:
        tournament.register_agent("Embeddings_Guesser", "guesser", GuesserEmbeddings)
    
    if NaiveGuesser:
        tournament.register_agent("Naive_Guesser", "guesser", NaiveGuesser)
    
    if GuesserSBERT:
        tournament.register_agent("SBERT_Guesser", "guesser", GuesserSBERT)
    
    if GuesserGPT:
        tournament.register_agent("GPT_Guesser", "guesser", GuesserGPT)
    
    if GuesserMCTS:
        tournament.register_agent("MCTS_Guesser", "guesser", GuesserMCTS, 
                                 num_simulations=50)  # Reduced for speed
    
    # Calculate tournament size
    num_cm = len(tournament.codemasters)
    num_g = len(tournament.guessers)
    total_teams = num_cm * num_g
    total_matchups = total_teams * (total_teams - 1)
    total_games = total_matchups * tournament.games_per_matchup
    
    print("\n" + "="*60)
    print("TOURNAMENT SETUP COMPLETE!")
    print("="*60)
    print(f"Codemasters registered: {num_cm}")
    print(f"Guessers registered: {num_g}")
    print(f"Total teams: {total_teams}")
    print(f"Total matchups: {total_matchups}")
    print(f"Total games: {total_games}")
    print(f"Estimated time: {total_games * 0.5 / 60:.1f} minutes")
    print()
    
    # Ask for confirmation before running large tournament
    if total_games > 200:
        choice = input("This is a large tournament. Continue? (y/n): ").lower().strip()
        if choice not in ['y', 'yes']:
            print("Tournament cancelled. Try smaller version below.")
            return
    
    # Run the tournament
    print("Starting tournament... This may take a while!")
    tournament.run_tournament()
    
    # Generate believability analysis
    print("\nGenerating believability analysis...")
    tournament.print_believability_analysis()
    tournament.save_believability_report()
    
    # Show composite rankings (performance + believability)
    composite_rankings = tournament.generate_composite_rankings()
    print("\n" + "="*80)
    print("FINAL COMPOSITE RANKINGS (Performance + Believability)")
    print("="*80)
    
    for i, (team, stats, believability, composite) in enumerate(composite_rankings[:15], 1):
        win_rate = stats.wins / max(1, stats.total_games) * 100
        print(f"{i:2d}. {team}")
        print(f"    Composite Score: {composite:.3f}")
        print(f"    Win Rate: {win_rate:.1f}% ({stats.wins}-{stats.losses})")
        print(f"    Believability: {believability:.3f}")
        print(f"    TrueSkill: {stats.trueskill_rating.mu:.2f} ± {stats.trueskill_rating.sigma:.2f}")
        print()

def run_focused_believability_test():
    """Run a focused test comparing key methods"""
    
    print("Setting up Focused Believability Test...")
    print("This will compare your new methods (CL, TOT) against baselines")
    
    tournament = BelievabilityTournament(
        tournament_name="Focused_Believability_Test",
        games_per_matchup=3  # More games for statistical significance
    )
    
    # Focus on key comparisons
    registered_agents = []
    
    if CodemasterEmbeddings:
        tournament.register_agent("Original_CM", "codemaster", CodemasterEmbeddings)
        registered_agents.append("Original_CM")
    
    if CodemasterCurriculum:
        tournament.register_agent("CL_CM", "codemaster", CodemasterCurriculum)
        registered_agents.append("CL_CM")
    
    if CodemasterTreeOfThoughts:
        tournament.register_agent("TOT_CM", "codemaster", CodemasterTreeOfThoughts)
        registered_agents.append("TOT_CM")
    
    if CodemasterMCTS:
        tournament.register_agent("MCTS_CM", "codemaster", CodemasterMCTS, 
                                 num_simulations=75)
        registered_agents.append("MCTS_CM")
    
    # Use reliable guessers
    guesser_count = 0
    if GuesserEmbeddings:
        tournament.register_agent("Embeddings_Guesser", "guesser", GuesserEmbeddings)
        guesser_count += 1
    
    if NaiveGuesser:
        tournament.register_agent("Naive_Guesser", "guesser", NaiveGuesser)
        guesser_count += 1
    
    if len(registered_agents) < 2 or guesser_count < 1:
        print(" Need at least 2 codemasters and 1 guesser for focused test")
        return
    
    print(f"\nFocused Test Setup:")
    print(f"- {len(registered_agents)} Codemasters × {guesser_count} Guessers = {len(registered_agents) * guesser_count} teams")
    print(f"- Focus: Compare CL and TOT against baseline methods")
    print(f"- Games per matchup: {tournament.games_per_matchup}")
    
    # Run tournament
    tournament.run_tournament()
    
    # Analysis
    tournament.print_believability_analysis()
    
    # Show results with focus on method comparison
    composite_rankings = tournament.generate_composite_rankings()
    print("\n" + "="*70)
    print("FOCUSED TEST RESULTS - METHOD COMPARISON")
    print("="*70)
    
    method_scores = {}
    for team, stats, believability, composite in composite_rankings:
        # Extract codemaster method
        cm_method = team.split('+')[0]  # Get codemaster name
        if cm_method not in method_scores:
            method_scores[cm_method] = {
                'teams': [],
                'believability_scores': [],
                'win_rates': []
            }
        
        win_rate = stats.wins / max(1, stats.total_games) * 100
        method_scores[cm_method]['teams'].append(team)
        method_scores[cm_method]['believability_scores'].append(believability)
        method_scores[cm_method]['win_rates'].append(win_rate)
    
    # Print method comparison
    print("\nMETHOD COMPARISON:")
    for method, data in method_scores.items():
        avg_believability = sum(data['believability_scores']) / len(data['believability_scores'])
        avg_win_rate = sum(data['win_rates']) / len(data['win_rates'])
        print(f"\n{method}:")
        print(f"  Average Believability: {avg_believability:.3f}")
        print(f"  Average Win Rate: {avg_win_rate:.1f}%")
        print(f"  Teams: {len(data['teams'])}")

def test_agent_imports():
    """Test if all agents can be imported correctly"""
    print("Testing agent imports...")
    
    agents_to_test = [
        ("CodemasterMCTS", CodemasterMCTS),
        ("CodemasterGPT", CodemasterGPT),
        ("CodemasterEmbeddings", CodemasterEmbeddings),
        ("CodemasterSBERT", CodemasterSBERT),
        ("CodemasterCurriculum", CodemasterCurriculum),
        ("CodemasterTreeOfThoughts", CodemasterTreeOfThoughts),
        ("GuesserEmbeddings", GuesserEmbeddings),
        ("NaiveGuesser", NaiveGuesser),
        ("GuesserSBERT", GuesserSBERT),
        ("GuesserGPT", GuesserGPT),
        ("GuesserMCTS", GuesserMCTS),
    ]
    
    success_count = 0
    failed_agents = []
    
    for name, agent_class in agents_to_test:
        if agent_class is not None:
            try:
                # Try to instantiate
                instance = agent_class()
                print(f" {name}: OK")
                success_count += 1
            except Exception as e:
                print(f" {name}: {e}")
                failed_agents.append(name)
        else:
            print(f"⚠️ {name}: Not imported")
            failed_agents.append(name)
    
    available_cm, available_g = count_available_agents()
    print(f"\n{success_count} agents working correctly")
    print(f"Available: {available_cm} codemasters, {available_g} guessers")
    
    if failed_agents:
        print(f"Issues with: {', '.join(failed_agents)}")
        return False
    else:
        print("🎉 All agents ready for tournament!")
        return True

if __name__ == "__main__":
    print("Codenames Believability Tournament")
    print("="*50)
    
    # Test imports first
    print("Checking agent availability...\n")
    imports_ok = test_agent_imports()
    
    available_cm, available_g = count_available_agents()
    
    if available_cm < 2 or available_g < 1:
        print(f"\n Insufficient agents: {available_cm} codemasters, {available_g} guessers")
        print("Need at least 2 codemasters and 1 guesser to run tournament.")
        print("Please fix import errors and try again.")
        exit(1)
    
    print(f"\n Ready to run tournament with {available_cm} codemasters and {available_g} guessers")
    
    print("\nChoose tournament type:")
    print("1. Focused test (compare CL/TOT vs baselines, ~50-100 games)")
    print("2. Complete tournament (all agents, many games)")
    print("3. Just test imports and exit")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    try:
        if choice == "1":
            run_focused_believability_test()
        elif choice == "2":
            run_complete_believability_tournament()
        elif choice == "3":
            print("Import test completed. Exiting.")
        else:
            print("Invalid choice. Running focused test...")
            run_focused_believability_test()
            
    except Exception as e:
        print(f" Tournament error: {e}")
        import traceback
        traceback.print_exc()
        print("\nCheck that all agent files exist and are properly formatted.")
        print("You may need to fix some agent implementations.")