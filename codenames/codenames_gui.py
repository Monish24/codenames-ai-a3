import tkinter as tk
from tkinter import scrolledtext, font, messagebox, ttk
import sys
import threading
import time
import subprocess
import queue
import re
import os
import json 
import csv   
from datetime import datetime  
from collections import defaultdict


# Enhanced tournament imports
try:
    from tournament import EnhancedTournamentManager
    from believability_tournament import EnhancedBelievabilityTournament
    ENHANCED_TOURNAMENT_AVAILABLE = True
except ImportError:
    ENHANCED_TOURNAMENT_AVAILABLE = False
    print("Enhanced tournament features not available")

# Try to import original tournament as fallback
try:
    from believability_tournament import BelievabilityTournament
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False

class CodenamesGUI:
    def __init__(self, root):
        # Game state
        self.board_buffer = ""
        self.root = root
        self.game_process = None
        self.game_running = False
        self.output_queue = queue.Queue()
        
        # Board state
        self.board_words = []
        self.key_grid = []
        self.word_results = []
        self.guessed_words = set()
        self.board_initialized = False
        self.spymaster_mode = True
        
        # FIXED: Game stats with proper reset handling
        self.red_words_found = 0
        self.blue_words_found = 0
        self.current_turn = "Red"
        self.max_red_words = 9
        self.max_blue_words = 8
        
        # Tournament state
        self.tournament_running = False
        self.current_game_number = 0
        self.total_tournament_games = 0
        
        # Enhanced data collection
        self.collect_enhanced_data = True
        self.current_game_clues = []
        self.current_game_guesses = []
        self.tournament_all_clues = []
        self.tournament_all_guesses = []
        
        # Enhanced metrics tracking
        self.agent_performance_tracker = {}
        
        self.setup_gui()
        self.setup_stdout_redirection()
        self.root.after(100, self.check_output_queue)
    
    def get_agent_performance(self, agent_name):
        """Helper to get agent performance with defaults"""
        if agent_name not in self.agent_performance_tracker:
            self.agent_performance_tracker[agent_name] = {
                'games_played': 0, 'wins': 0, 'total_clues': 0, 'total_guesses': 0,
                'clue_efficiency': [], 'guess_accuracy': []
            }
        return self.agent_performance_tracker[agent_name]
    
    def setup_gui(self):
        """Setup the complete GUI"""
        self.root.title("Enhanced Codenames AI Game & Tournament")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2C3E50")
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#242731")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(main_frame, text="Enhanced Codenames AI Game & Tournament", 
                font=("Helvetica", 22, "bold"), bg="#242731", fg="#E7E1BD").pack(pady=10)
        
        # Control buttons
        self.create_enhanced_controls(main_frame)
        
        # Team configuration
        self.create_team_config(main_frame)
        
        # Game area (board + info panel)
        self.create_game_area(main_frame)
    
    def create_enhanced_controls(self, parent):
        """Enhanced control buttons"""
        controls = tk.Frame(parent, bg="#242731")
        controls.pack(fill=tk.X, pady=5)
        
        button_config = {"font": ("Helvetica", 12), "relief": tk.FLAT, "bd": 0, "padx": 10}
        
        # Existing buttons
        self.start_button = tk.Button(controls, text="Start Game", command=self.start_game, 
                                     bg="#4CAF50", fg="black", width=15, **button_config)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(controls, text="Stop Game", command=self.stop_game,
                                    bg="#F44336", fg="black", width=15, state=tk.DISABLED, **button_config)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Enhanced tournament button
        tournament_state = tk.NORMAL if ENHANCED_TOURNAMENT_AVAILABLE else tk.DISABLED
        self.tournament_button = tk.Button(controls, text="Enhanced Tournament", 
                                          command=self.open_enhanced_tournament,
                                          bg="#9C27B0", fg="black", width=18, 
                                          state=tournament_state, **button_config)
        self.tournament_button.pack(side=tk.LEFT, padx=5)
        
        self.spymaster_button = tk.Button(controls, text="Spymaster View", command=self.toggle_view,
                                         bg="#673AB7", fg="black", width=18, **button_config)
        self.spymaster_button.pack(side=tk.LEFT, padx=5)
    
    def create_team_config(self, parent):
        """Create team configuration dropdowns"""
        config_frame = tk.Frame(parent, bg="#242731")
        config_frame.pack(fill=tk.X, pady=5)
        
        agents = ["MCTS", "EMD", "SBERT", "CL", "TOT", "Naive"]
        
        # Red team
        red_frame = tk.Frame(config_frame, bg="#242731")
        red_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(red_frame, text="Red Team:", font=("Helvetica", 12, "bold"), 
                bg="#242731", fg="#C13A37").pack(anchor=tk.W)
        
        self.red_codemaster = tk.StringVar(value="MCTS")
        self.red_guesser = tk.StringVar(value="EMD")
        self.create_role_dropdown(red_frame, "Codemaster:", self.red_codemaster, agents)
        self.create_role_dropdown(red_frame, "Guesser:", self.red_guesser, agents)
        
        # Blue team
        blue_frame = tk.Frame(config_frame, bg="#242731")
        blue_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(blue_frame, text="Blue Team:", font=("Helvetica", 12, "bold"),
                bg="#242731", fg="#4989C5").pack(anchor=tk.W)
        
        self.blue_codemaster = tk.StringVar(value="EMD")
        self.blue_guesser = tk.StringVar(value="EMD")
        self.create_role_dropdown(blue_frame, "Codemaster:", self.blue_codemaster, agents)
        self.create_role_dropdown(blue_frame, "Guesser:", self.blue_guesser, agents)
        
        # Seed input
        tk.Label(config_frame, text="Seed:", font=("Helvetica", 12, "bold"), 
                bg="#242731", fg="#E7E1BD").pack(side=tk.LEFT, padx=5)
        self.seed_entry = tk.Entry(config_frame, font=("Helvetica", 12), width=10,
                                  bg="#34495E", fg="white", insertbackground="white")
        self.seed_entry.insert(0, "42")
        self.seed_entry.pack(side=tk.LEFT, padx=5)
    
    def create_role_dropdown(self, parent, label, var, options):
        """Helper to create role dropdown"""
        frame = tk.Frame(parent, bg="#242731")
        frame.pack(fill=tk.X, pady=2)
        tk.Label(frame, text=label, font=("Helvetica", 12), bg="#242731", fg="#E7E1BD").pack(side=tk.LEFT, padx=2)
        dropdown = tk.OptionMenu(frame, var, *options)
        dropdown.config(font=("Helvetica", 12), bg="#34495E", fg="white", width=12)
        dropdown.pack(side=tk.LEFT, padx=2)
    
    def create_game_area(self, parent):
        """Create the main game area with board and info panel"""
        content_paned = tk.PanedWindow(parent, bg="#242731", orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Board frame
        board_frame = tk.Frame(content_paned, bg="#1E1F29", bd=0)
        self.create_board(board_frame)
        
        # Info frame  
        info_frame = tk.Frame(content_paned, bg="#1E1F29", bd=0)
        self.create_enhanced_info_panel(info_frame)
        
        content_paned.add(board_frame, stretch="always", width=600)
        content_paned.add(info_frame, stretch="always", width=400)
    
    def create_board(self, parent):
        """Create the game board"""
        # Board title
        title_frame = tk.Frame(parent, bg="#1E1F29")
        title_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(title_frame, text="Game Board", font=("Helvetica", 18, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(side=tk.LEFT, padx=10)
        
        self.turn_label = tk.Label(title_frame, text="", font=("Helvetica", 12, "bold"),
                                  bg="#1E1F29", fg="#F1C40F")
        self.turn_label.pack(side=tk.RIGHT, padx=10)
        
        # Board grid
        board_grid = tk.Frame(parent, bg="#1E1F29", padx=15, pady=15)
        board_grid.pack(fill=tk.BOTH, expand=True)
        
        self.word_tiles = []
        for i in range(5):
            row = []
            for j in range(5):
                tile_frame = tk.Frame(board_grid, bg="#1E1F29", padx=3, pady=3)
                tile_frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                
                tile = tk.Label(tile_frame, text="Loading...", width=10, height=3,
                               font=("Helvetica", 12, "bold"), bg="#E7E1BD", fg="black",
                               relief=tk.RAISED, bd=2)
                tile.pack(fill=tk.BOTH, expand=True)
                row.append(tile)
            self.word_tiles.append(row)
        
        # Configure grid weights
        for i in range(5):
            board_grid.grid_rowconfigure(i, weight=1)
            board_grid.grid_columnconfigure(i, weight=1)
    
    def create_enhanced_info_panel(self, parent):
        """Enhanced info panel with metrics"""
        # Current clue section (enhanced)
        clue_frame = tk.Frame(parent, bg="#1E1F29", bd=1, relief=tk.GROOVE)
        clue_frame.pack(fill=tk.X, pady=5, padx=10)
        
        clue_header = tk.Frame(clue_frame, bg="#1E1F29")
        clue_header.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(clue_header, text="Current Clue:", font=("Helvetica", 12, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(side=tk.LEFT)
        
        self.clue_label = tk.Label(clue_frame, text="Waiting for clue...", 
                                  font=("Helvetica", 18, "bold"), bg="#1E1F29", fg="#F1C40F")
        self.clue_label.pack(padx=10, pady=5)
        
        # Enhanced guess section
        guess_frame = tk.Frame(parent, bg="#1E1F29", bd=1, relief=tk.GROOVE)
        guess_frame.pack(fill=tk.X, pady=5, padx=10)
        
        guess_header = tk.Frame(guess_frame, bg="#1E1F29")
        guess_header.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(guess_header, text="Latest Guess:", font=("Helvetica", 12, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(side=tk.LEFT)
        
        self.guess_label = tk.Label(guess_frame, text="Waiting for guess...",
                                   font=("Helvetica", 18, "bold"), bg="#1E1F29", fg="white")
        self.guess_label.pack(padx=10, pady=5)
        
        # Enhanced score display
        self.create_enhanced_score_display(parent)
        
        # Game log
        self.create_game_log(parent)
    
    def create_enhanced_score_display(self, parent):
        """Enhanced score display with proper reset handling"""
        score_frame = tk.Frame(parent, bg="#1E1F29")
        score_frame.pack(fill=tk.X, pady=10, padx=10)
        
        tk.Label(score_frame, text="Score:", font=("Helvetica", 12, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(anchor=tk.W, pady=5)
        
        score_display = tk.Frame(score_frame, bg="#1E1F29")
        score_display.pack(fill=tk.X, pady=5)
        
        # Red score
        red_frame = tk.Frame(score_display, bg="#1E1F29")
        red_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(red_frame, text="Red", font=("Helvetica", 14, "bold"),
                bg="#1E1F29", fg="#C13A37").pack()
        self.red_score = tk.Label(red_frame, text="0/9", font=("Helvetica", 20, "bold"),
                                 bg="#1E1F29", fg="#C13A37")
        self.red_score.pack()
        
        # Blue score
        blue_frame = tk.Frame(score_display, bg="#1E1F29")
        blue_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        tk.Label(blue_frame, text="Blue", font=("Helvetica", 14, "bold"),
                bg="#1E1F29", fg="#4989C5").pack()
        self.blue_score = tk.Label(blue_frame, text="0/8", font=("Helvetica", 20, "bold"),
                                  bg="#1E1F29", fg="#4989C5")
        self.blue_score.pack()
    
    def create_game_log(self, parent):
        """Create game log"""
        log_frame = tk.Frame(parent, bg="#1E1F29")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)
        
        tk.Label(log_frame, text="Game History", font=("Helvetica", 18, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, font=("Helvetica", 11),
                                                 bg="#242731", fg="#E7E1BD", relief=tk.FLAT, bd=1,
                                                 padx=10, pady=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text tags
        tags = {
            "red_turn": {"foreground": "#C13A37", "font": ("Helvetica", 12, "bold")},
            "blue_turn": {"foreground": "#4989C5", "font": ("Helvetica", 12, "bold")},
            "red_clue": {"foreground": "#C13A37"},
            "blue_clue": {"foreground": "#4989C5"},
            "red_guess": {"foreground": "#C13A37"},
            "blue_guess": {"foreground": "#4989C5"},
            "game_event": {"foreground": "#E7E1BD", "font": ("Helvetica", 12, "bold")},
            "win": {"foreground": "#FFD700", "font": ("Helvetica", 16, "bold")},
            "debug": {"foreground": "#E67E22"},
            "tournament": {"foreground": "#9C27B0", "font": ("Helvetica", 12, "bold")}
        }
        
        for tag, config in tags.items():
            self.log_text.tag_configure(tag, **config)
    
    # ==================== GAME LOGIC ====================
    
    def start_game(self):
        """Start a new game"""
        if self.game_running:
            return
        
        self.reset_game_state()
        self.game_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Build command
        cmd = self.build_game_command()
        self.add_log_entry(f"Starting game: {' '.join(cmd[:5])}...", "debug")
        
        # Start game in thread
        threading.Thread(target=self.run_game, args=(cmd,), daemon=True).start()
    
    def build_game_command(self):
        """Build the game command"""
        agent_map = {
            "MCTS": {"cm": "players.codemasterMCTS.CodemasterMCTS", "g": "players.guesser_MCTS.GuesserMCTS"},
            "EMD": {"cm": "players.codemaster_EMD.CodemasterEmbeddings", "g": "players.guesserEMD.GuesserEmbeddings"},
            "SBERT": {"cm": "players.codemaster_SBERT.CodemasterSBERT", "g": "players.guesser_SBERT.GuesserSBERT"},
            "CL": {"cm": "players.codemaster_CL.CodemasterCurriculum", "g": "players.guesserEMD.GuesserEmbeddings"},
            "TOT": {"cm": "players.codemaster_TOT.CodemasterTreeOfThoughts", "g": "players.guesserEMD.GuesserEmbeddings"},
            "Naive": {"cm": "players.codemasterMCTS.CodemasterMCTS", "g": "players.guesser_naive.NaiveGuesser"},
        }
        
        def get_agent(agent_type, role):
            return agent_map.get(agent_type, agent_map["EMD"])[role]
        
        return [
            "python", "run_game.py",
            get_agent(self.red_codemaster.get(), "cm"),
            get_agent(self.red_guesser.get(), "g"),
            get_agent(self.blue_codemaster.get(), "cm"),
            get_agent(self.blue_guesser.get(), "g"),
            "--seed", self.seed_entry.get() or "42"
        ]
    
    def run_game(self, cmd):
        """Run the game process"""
        try:
            self.game_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1, universal_newlines=True)
            
            for line in iter(self.game_process.stdout.readline, ''):
                self.output_queue.put(line)
            
            self.game_process.wait()
        except Exception as e:
            self.output_queue.put(f"Error: {str(e)}\n")
        finally:
            self.root.after(0, self.reset_game_controls)
    
    def stop_game(self):
        """Stop the running game"""
        if self.game_process:
            self.game_process.terminate()
            self.reset_game_controls()
    
    def reset_game_state(self):
        """FIXED: Proper game state reset for tournaments"""
        # Reset board state
        self.board_words = []
        self.word_results = []
        self.guessed_words = set()
        self.board_initialized = False
        
        # FIXED: Reset scores properly
        self.red_words_found = 0
        self.blue_words_found = 0
        self.current_turn = "Red"
        
        # Reset display elements
        self.clue_label.config(text="Waiting for clue...")
        self.guess_label.config(text="Waiting for guess...")
        self.red_score.config(text="0/9")
        self.blue_score.config(text="0/8")
        self.turn_label.config(text="")
        
        # FIXED: Only clear log if not in tournament
        if not self.tournament_running:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        # FIXED: Reset board tiles properly
        for row in self.word_tiles:
            for tile in row:
                tile.config(text="Loading...", bg="#E7E1BD", fg="#2C3E50", 
                           relief=tk.RAISED, bd=2)
        
        # Reset enhanced data collection
        self.current_game_clues = []
        self.current_game_guesses = []
    
    def reset_game_controls(self):
        """Reset game controls after game ends"""
        self.game_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if not self.tournament_running:
            self.add_log_entry("Game Ended", "game_event")
    
    # ==================== OUTPUT PROCESSING ====================
    
    def setup_stdout_redirection(self):
        """Setup stdout redirection"""
        self.original_stdout = sys.stdout
        
        class StdoutRedirector:
            def __init__(self, queue):
                self.queue = queue
            def write(self, string):
                self.queue.put(string)
            def flush(self):
                pass
        
        sys.stdout = StdoutRedirector(self.output_queue)
    
    def check_output_queue(self):
        """Check for new output"""
        try:
            while not self.output_queue.empty():
                output = self.output_queue.get_nowait()
                self.process_enhanced_output(output)
        except:
            pass
        self.root.after(100, self.check_output_queue)
    
    def process_enhanced_output(self, output):
        """Enhanced output processing"""
        if not output or not output.strip():
            return
        self.board_buffer += output
        clean = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
        
        # FIXED: Board extraction with proper reset and game numbering
        if (not self.board_initialized and
            "____________________________KEY"   in self.board_buffer and
            "___________________________BOARD" in self.board_buffer):

            if self.try_extract_board(self.board_buffer):
                self.board_buffer = ""
                if self.tournament_running:
                    self.current_game_number += 1
                    self.add_log_entry(f"🎮 Game {self.current_game_number}: Board initialized", "debug")
            return
        
        # Team turns with tournament game numbering
        if "RED TEAM TURN" in clean:
            self.current_turn = "Red"
            self.turn_label.config(text="Red Team's Turn", fg="#C13A37")
            log_text = "Red Team's Turn"
            if self.tournament_running:
                log_text = f"Game {self.current_game_number}: {log_text}"
            self.add_log_entry(log_text, "red_turn")
        elif "BLUE TEAM TURN" in clean:
            self.current_turn = "Blue"
            self.turn_label.config(text="Blue Team's Turn", fg="#4989C5")
            log_text = "Blue Team's Turn"
            if self.tournament_running:
                log_text = f"Game {self.current_game_number}: {log_text}"
            self.add_log_entry(log_text, "blue_turn")
        
        # Enhanced clue detection
        clue_patterns = [
            r"STRUCTURED_CLUE:\s*([^|]+)\|([^|]+)\|(\d+)\|([^|]+)",
            r"The clue is:\s*(\w+)\s+(\d+)",
            r"clue is:\s*(\w+)\s+(\d+)"
        ]
        
        for pattern in clue_patterns:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                if "STRUCTURED_CLUE" in pattern:
                    codemaster_name, clue_word, clue_num, team = match.groups()
                else:
                    clue_word, clue_num = match.groups()
                    team = self.current_turn
                    codemaster_name = "Unknown"
                
                clue_text = f"{clue_word.strip().upper()} ({clue_num})"
                team_color = "#E74C3C" if "RED" in team.upper() or self.current_turn == "Red" else "#3498DB"
                
                self.clue_label.config(text=clue_text, fg=team_color)
                
                # Enhanced clue logging
                log_text = f"Clue: {clue_word} {clue_num}"
                if self.tournament_running:
                    log_text = f"Game {self.current_game_number}: {log_text} ({codemaster_name})"
                
                log_color = "red_clue" if self.current_turn == "Red" else "blue_clue"
                self.add_log_entry(log_text, log_color)
                return
        
        # Enhanced guess detection
        guess_patterns = [
            r"STRUCTURED_GUESS:\s*([^|]+)\|([^|]+)",
            r"Guessing:\s*([A-Z]+)",
            r"Selected:\s*([A-Z]+)",
            r"Guesser selected:\s*([A-Z]+)"
        ]
        
        for pattern in guess_patterns:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                if "STRUCTURED_GUESS" in pattern:
                    team, word = match.groups()
                    guess_word = word.upper()
                else:
                    guess_word = match.group(1).upper()
                    team = self.current_turn
                
                self.guess_label.config(text=f"{guess_word} ⏳")
                color = "red_guess" if "RED" in team.upper() or self.current_turn == "Red" else "blue_guess"
                
                log_text = f"Guess: {guess_word}"
                if self.tournament_running:
                    log_text = f"Game {self.current_game_number}: {log_text}"
                
                self.add_log_entry(log_text, color)
                return
        
        # FIXED: Enhanced result detection with proper score tracking
        if "GUESS_RESULT:" in clean:
            parts = clean.split("GUESS_RESULT:")[1].strip().split("|")
            if len(parts) >= 3:
                word, team_type, turn = parts[:3]
                self.process_enhanced_guess_result(word, team_type, turn)
        
        # Win conditions
        elif "GAME_END:" in clean or "Red Team Wins" in clean or "Blue Team Wins" in clean:
            if "Red" in clean:
                win_text = "RED TEAM WINS!"
                if self.tournament_running:
                    win_text = f"Game {self.current_game_number}: {win_text}"
                self.add_log_entry(win_text, "win")
                self.turn_label.config(text="Red Team Wins!", fg="#C13A37")
            else:
                win_text = "BLUE TEAM WINS!"
                if self.tournament_running:
                    win_text = f"Game {self.current_game_number}: {win_text}"
                self.add_log_entry(win_text, "win")
                self.turn_label.config(text="Blue Team Wins!", fg="#4989C5")
            
            # Collect game data if in tournament
            if self.tournament_running:
                self.collect_game_data()
        
        # Seed info
        elif "seed:" in clean.lower():
            seed_text = f"ℹ️ {clean}"
            if self.tournament_running:
                seed_text = f"Game {self.current_game_number}: {seed_text}"
            self.add_log_entry(seed_text, "debug")

    def process_enhanced_guess_result(self, word, team_type, current_turn):
        """FIXED: Enhanced guess result processing with proper score tracking"""
        word = word.upper()
        team_type = team_type.lower()
        is_correct = (team_type == "red" and "RED" in current_turn) or (team_type == "blue" and "BLUE" in current_turn)
        
        # FIXED: Proper score tracking
        if team_type == "red":
            emoji = "✅" if is_correct else "❌"
            result = f"{'Correct' if is_correct else 'Wrong'}: {word} is a Red Card"
            if is_correct and "RED" in current_turn:  # FIXED: Only increment for correct team
                self.red_words_found += 1
                self.red_score.config(text=f"{self.red_words_found}/9")
        elif team_type == "blue":
            emoji = "✅" if is_correct else "❌"
            result = f"{'Correct' if is_correct else 'Wrong'}: {word} is a Blue Card"
            if is_correct and "BLUE" in current_turn:  # FIXED: Only increment for correct team
                self.blue_words_found += 1
                self.blue_score.config(text=f"{self.blue_words_found}/8")
        elif team_type == "civilian":
            emoji = "⚪"
            result = f"Neutral: {word} is a Civilian Card"
        elif team_type == "assassin":
            emoji = "☠️"
            result = f"ASSASSIN: {word} is the Assassin! (Game Over)"
        else:
            emoji = "❓"
            result = f"Unknown: {word}"
        # Update display
        self.guess_label.config(text=f"{word} {emoji}")
        color = "red_guess" if "RED" in current_turn else "blue_guess"
        
        log_text = result
        if self.tournament_running:
            log_text = f"Game {self.current_game_number}: {result}"
        
        self.add_log_entry(log_text, color)
        
        # Update tile and tracking
        self.update_tile(word, team_type)
        self.guessed_words.add(word)
    def collect_game_data(self):
        """Collect data from completed game for tournament analysis"""
        if not self.tournament_running:
            return
        
        # Store current game data
        game_data = {
            'game_number': self.current_game_number,
            'red_score': self.red_words_found,
            'blue_score': self.blue_words_found,
            'total_guesses': len(self.guessed_words),
            'clues': self.current_game_clues.copy(),
            'guesses': self.current_game_guesses.copy()
        }
        
        # Add to tournament data
        self.tournament_all_clues.extend(self.current_game_clues)
        self.tournament_all_guesses.extend(self.current_game_guesses)
        
        self.add_log_entry(f"📊 Game {self.current_game_number} data collected", "debug")
    
    # ==================== BOARD MANAGEMENT ====================
    
    def try_extract_board(self, output):
        """Extract board from game output - FIXED VERSION"""
        if self.board_initialized:
            return True
        
        # Method 1: Look for the complete KEY + BOARD pattern
        key_pattern = r"____________________________KEY____________________________"
        board_pattern = r"___________________________BOARD___________________________"
        
        if key_pattern in output and board_pattern in output:
            key_start = output.find(key_pattern)
            board_start = output.find(board_pattern)
            
            if key_start >= 0 and board_start >= 0:
                # Extract sections
                key_section = output[key_start:board_start]
                board_section = output[board_start:board_start + 1000]
                
                # Parse key section for team assignments
                key_teams = []
                for line in key_section.split('\n'):
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    teams = re.findall(r'\b(Red|Blue|Civilian|Assassin)\b', clean_line)
                    key_teams.extend([t.lower() for t in teams])
                
                # Parse board section for words
                board_words = []
                for line in board_section.split('\n'):
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    words = re.findall(r'\b[A-Z]{2,}\b', clean_line)
                    filtered_words = [w for w in words if w not in ['RED', 'BLUE', 'CIVILIAN', 'ASSASSIN', 'KEY', 'BOARD']]
                    board_words.extend(filtered_words)
                
                if len(key_teams) >= 25 and len(board_words) >= 25:
                    self.word_results = [(board_words[i], key_teams[i]) for i in range(25)]
                    self.initialize_board(board_words[:25])
                    return True
        
        return False    
    
    def initialize_board(self, words):
        """Initialize the board with words"""
        if self.board_initialized:
            return
        
        self.board_words = words[:25]
        for i in range(5):
            for j in range(5):
                self.word_tiles[i][j].config(text=words[i*5 + j], bg="#E7E1BD", fg="black")
        
        self.board_initialized = True
        self.update_board_display()
        if not self.tournament_running:
            self.add_log_entry("Game Started", "game_event")
    
    def update_tile(self, word, team_type):
        """Update a specific tile"""
        colors = {"red": "#C13A37", "blue": "#4989C5", "civilian": "#DCD6B0", "assassin": "#2C2C2E"}
        
        for row in self.word_tiles:
            for tile in row:
                if tile.cget("text").upper() == word:
                    tile.config(bg=colors.get(team_type, "#999999"), fg="white", relief=tk.SUNKEN, bd=3)
                    break
        
        if not self.spymaster_mode:
            self.update_board_display()
    
    def update_board_display(self):
        """Update board display based on spymaster mode"""
        if not self.board_initialized:
            return
        
        colors = {"red": "#C13A37", "blue": "#4989C5", "civilian": "#DCD6B0", "assassin": "#2C2C2E"}
        
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                word = tile.cget("text").upper()
                
                # Find team for this word
                team = next((t for w, t in self.word_results if w.upper() == word), "unknown")
                is_guessed = word in self.guessed_words
                
                if self.spymaster_mode:
                    # Show all team colors
                    tile.config(bg=colors.get(team, "#E7E1BD"), 
                               fg="white" if team in colors else "black",
                               relief=tk.SUNKEN if is_guessed else tk.RAISED, bd=3 if is_guessed else 2)
                else:
                    # Only show colors for guessed cards
                    if is_guessed:
                        tile.config(bg=colors.get(team, "#999999"), fg="white", relief=tk.SUNKEN, bd=3)
                    else:
                        tile.config(bg="#E7E1BD", fg="black", relief=tk.RAISED, bd=2)
    
    def toggle_view(self):
        """Toggle between spymaster and player view"""
        if not self.board_initialized:
            return
        
        self.spymaster_mode = not self.spymaster_mode
        self.spymaster_button.config(text="Player View" if self.spymaster_mode else "Spymaster View")
        self.update_board_display()
        view_text = f"Switched to {'Spymaster' if self.spymaster_mode else 'Player'} View"
        if self.tournament_running:
            view_text = f"Game {self.current_game_number}: {view_text}"
        self.add_log_entry(view_text, "debug")
    
    # ==================== UTILITY METHODS ====================
    
    def add_log_entry(self, text, tag=None):
        """Add entry to game log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    # ==================== ENHANCED TOURNAMENT METHODS ====================
    
    def open_enhanced_tournament(self):
        """Open enhanced tournament settings"""
        if not ENHANCED_TOURNAMENT_AVAILABLE:
            messagebox.showerror("Enhanced Tournament Unavailable", 
                                "Enhanced tournament system not available.\nPlease check that all enhanced tournament files are present.")
            return
        
        if self.tournament_running:
            messagebox.showwarning("Tournament Running", 
                                  "A tournament is already running. Please wait for it to complete.")
            return
        
        settings_window = EnhancedTournamentSettingsWindow(self)
        self.root.wait_window(settings_window.window)
        
        if settings_window.result:
            self.start_enhanced_tournament(settings_window.result)

    def start_enhanced_tournament(self, config):
        """Start enhanced tournament with believability analysis"""
        self.clear_tournament_data()
        self.tournament_running = True
        self.current_game_number = 0
        self.tournament_button.config(state=tk.DISABLED)
        
        # Show enhanced progress window
        self.tournament_progress_window = EnhancedTournamentProgressWindow(self, config)
        
        # Start tournament in background thread
        self.tournament_thread = threading.Thread(target=self.run_enhanced_tournament, args=(config,))
        self.tournament_thread.daemon = True
        self.tournament_thread.start()
        
        self.add_log_entry("🏆 Enhanced Tournament Started!", "tournament")

    def run_enhanced_tournament(self, config):
        """Run enhanced tournament with comprehensive analysis"""
        try:
            # Create appropriate tournament instance
            if config.get('believability_analysis', False):
                tournament = EnhancedBelievabilityTournament(
                    tournament_name=f"GUI_Enhanced_Tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    games_per_matchup=config['games_per_matchup'],
                    max_matchups=config.get('max_games', 200) // config['games_per_matchup']
                )
            else:
                tournament = EnhancedTournamentManager(
                    tournament_name=f"GUI_Performance_Tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    games_per_matchup=config['games_per_matchup'],
                    max_matchups=config.get('max_games', 200) // config['games_per_matchup']
                )
            
            # Agent registration
            self.register_tournament_agents(tournament, config)
            
            # Calculate tournament size
            total_matchups = len(tournament.generate_matchups())
            total_games = total_matchups * config['games_per_matchup']
            self.total_tournament_games = total_games
            
            self.update_tournament_progress(f"Starting {total_games} games with enhanced analysis...")
            
            # Run tournament with progress tracking
            games_completed = 0
            
            def track_enhanced_progress():
                nonlocal games_completed
                while self.tournament_running and games_completed < total_games:
                    current_completed = len(tournament.match_results)
                    if current_completed > games_completed:
                        games_completed = current_completed
                        
                        # Update progress window
                        if (self.tournament_progress_window and 
                            not self.tournament_progress_window.cancelled):
                            
                            current_match = ""
                            if tournament.match_results:
                                recent = tournament.match_results[-1]
                                current_match = (f"{recent.red_codemaster}+{recent.red_guesser} "
                                               f"vs {recent.blue_codemaster}+{recent.blue_guesser}")
                            
                            self.root.after(0, lambda: self.tournament_progress_window.update_enhanced_progress(
                                games_completed, total_games, current_match, "Playing Games"
                            ))
                        
                        # Check for cancellation
                        if (self.tournament_progress_window and 
                            self.tournament_progress_window.cancelled):
                            raise Exception("Tournament cancelled by user")
                    
                    time.sleep(1)
            
            # Start progress tracking
            progress_thread = threading.Thread(target=track_enhanced_progress, daemon=True)
            progress_thread.start()
            
            # Run the tournament
            if config.get('believability_analysis', False):
                results = tournament.run_tournament_with_believability(shuffle_matchups=True)
            else:
                results = tournament.run_tournament(shuffle_matchups=True)
            
            # Show results
            self.root.after(0, lambda: self.show_enhanced_tournament_results(results))
            
        except Exception as e:
            error_msg = f"Enhanced tournament error: {str(e)}"
            print(f"DEBUG: {error_msg}")
            if self.tournament_progress_window:
                self.root.after(0, lambda: self.tournament_progress_window.add_analysis_log(error_msg, "Error"))
            self.root.after(0, lambda: messagebox.showerror("Tournament Error", error_msg))
        
        finally:
            self.root.after(0, self.enhanced_tournament_finished)

    def register_tournament_agents(self, tournament, config):
        """Register agents for enhanced tournament"""
        # Agent mapping
        agent_map = {
            'MCTS': {
                'codemaster': ('players.codemasterMCTS', 'CodemasterMCTS'),
                'guesser': ('players.guesser_MCTS', 'GuesserMCTS')
            },
            'EMD': {
                'codemaster': ('players.codemaster_EMD', 'CodemasterEmbeddings'),
                'guesser': ('players.guesserEMD', 'GuesserEmbeddings')
            },
            'SBERT': {
                'codemaster': ('players.codemaster_SBERT', 'CodemasterSBERT'),
                'guesser': ('players.guesser_SBERT', 'GuesserSBERT')
            },
            'CL': {
                'codemaster': ('players.codemaster_CL', 'CodemasterCurriculum'),
                'guesser': ('players.guesser_naive', 'NaiveGuesser')
            },
            'TOT': {
                'codemaster': ('players.codemaster_TOT', 'CodemasterTreeOfThoughts'),
                'guesser': ('players.guesser_naive', 'NaiveGuesser')
            },
            'Naive': {
                'codemaster': ('players.codemaster_CL', 'CodemasterCurriculum'),
                'guesser': ('players.guesser_naive', 'NaiveGuesser')
            }
        }
        
        registered_count = 0
        
        # Register codemasters
        for cm_code in config['codemasters']:
            if cm_code in agent_map and agent_map[cm_code]['codemaster']:
                try:
                    module_path, class_name = agent_map[cm_code]['codemaster']
                    module = __import__(module_path, fromlist=[class_name])
                    agent_class = getattr(module, class_name)
                    tournament.register_agent(f"{cm_code}_CM", "codemaster", agent_class)
                    registered_count += 1
                    self.update_tournament_progress(f"Registered {cm_code} codemaster")
                except Exception as e:
                    self.update_tournament_progress(f"Failed to register {cm_code} codemaster: {e}")
        
        # Register guessers
        for g_code in config['guessers']:
            if g_code in agent_map and agent_map[g_code]['guesser']:
                try:
                    module_path, class_name = agent_map[g_code]['guesser']
                    module = __import__(module_path, fromlist=[class_name])
                    agent_class = getattr(module, class_name)
                    tournament.register_agent(f"{g_code}_Guesser", "guesser", agent_class)
                    registered_count += 1
                    self.update_tournament_progress(f"Registered {g_code} guesser")
                except Exception as e:
                    self.update_tournament_progress(f"Failed to register {g_code} guesser: {e}")
        
        return registered_count

    def update_tournament_progress(self, message):
        """Update tournament progress (called from background thread)"""
        if self.tournament_progress_window:
            self.root.after(0, lambda: self.tournament_progress_window.add_analysis_log(message, "Info"))

    def show_enhanced_tournament_results(self, results):
        """Show enhanced tournament results window"""
        if self.tournament_progress_window:
            self.tournament_progress_window.close()
        
        EnhancedTournamentResultsWindow(self, results)
        
        # Add summary to main log
        self.add_log_entry("🏆 Enhanced Tournament Completed!", "tournament")
        self.add_log_entry(f"Total games: {results.total_games}", "tournament")
        
        if hasattr(results, 'team_rankings') and results.team_rankings:
            winner = results.team_rankings[0]
            team_name = winner[0] if isinstance(winner, tuple) else str(winner)
            self.add_log_entry(f"Winner: {team_name}", "tournament")
        
        # Show believability summary if available
        if hasattr(results, 'believability_analysis') and results.believability_analysis:
            analysis = results.believability_analysis
            if 'top_believable_codemasters' in analysis:
                top_believable = analysis['top_believable_codemasters']
                if top_believable:
                    top_name, top_score = top_believable[0]
                    self.add_log_entry(f"Most Believable: {top_name} ({top_score:.3f})", "tournament")

    def clear_tournament_data(self):
        """Clear all tournament data for fresh start"""
        self.tournament_all_clues = []
        self.tournament_all_guesses = []
        self.current_game_clues = []
        self.current_game_guesses = []
        self.current_game_number = 0
        self.total_tournament_games = 0
        self.agent_performance_tracker = {}

    def enhanced_tournament_finished(self):
        """Clean up after enhanced tournament finishes"""
        self.tournament_running = False
        self.tournament_button.config(state=tk.NORMAL)
        self.tournament_thread = None
        self.current_game_number = 0
        
        if self.tournament_progress_window:
            self.tournament_progress_window.close()
            self.tournament_progress_window = None
        
        self.add_log_entry("🏆 Enhanced Tournament Completed!", "tournament")

# ==================== ENHANCED TOURNAMENT SETTINGS WINDOW ====================

class EnhancedTournamentSettingsWindow:
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        
        # Create modal window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Enhanced Tournament Settings")
        self.window.geometry("700x800")
        self.window.configure(bg="#242731")
        self.window.transient(parent.root)
        self.window.grab_set()
        
        # Center the window
        self.window.geometry("+{}+{}".format(
            parent.root.winfo_rootx() + 50,
            parent.root.winfo_rooty() + 50
        ))
        
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.window, bg="#242731")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Enhanced Tournament Configuration", 
                              font=("Helvetica", 18, "bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Tournament Type Tab
        type_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(type_frame, text="Tournament Type")
        
        # Agent Selection Tab
        agent_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(agent_frame, text="Agent Selection")
        
        # Enhanced Settings Tab
        settings_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(settings_frame, text="Enhanced Settings")
        
        # Setup tabs
        self.setup_tournament_type_tab(type_frame)
        self.setup_agent_selection_tab(agent_frame)
        self.setup_enhanced_settings_tab(settings_frame)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#242731")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Cancel button
        cancel_btn = tk.Button(button_frame, text="Cancel", command=self.cancel,
                              font=("Helvetica", 12),
                              bg="#F44336", fg="white", relief=tk.FLAT, bd=0, 
                              padx=20, pady=8)
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Start tournament button
        start_btn = tk.Button(button_frame, text="Start Enhanced Tournament", command=self.start_tournament,
                             font=("Helvetica", 12, "bold"),
                             bg="#4CAF50", fg="black", relief=tk.FLAT, bd=0, 
                             padx=20, pady=8)
        start_btn.pack(side=tk.RIGHT)
    
    def setup_tournament_type_tab(self, parent):
        """Setup tournament type selection"""
        # Tournament type selection
        type_label = tk.Label(parent, text="Tournament Type:", 
                             font=("Helvetica", 14, "bold"),
                             bg="#242731", fg="#E7E1BD")
        type_label.pack(anchor=tk.W, pady=(20, 10), padx=20)
        
        self.tournament_type = tk.StringVar(value="performance")
        
        performance_rb = tk.Radiobutton(parent, text="Performance Tournament", 
                                       variable=self.tournament_type, value="performance",
                                       font=("Helvetica", 12),
                                       bg="#242731", fg="#E7E1BD", selectcolor="#242731",
                                       activebackground="#242731", activeforeground="#E7E1BD",
                                       command=self.update_tournament_info)
        performance_rb.pack(anchor=tk.W, pady=5, padx=40)
        
        perf_desc = tk.Label(parent, text="Focus on win rates, TrueSkill ratings, and performance metrics",
                            font=("Helvetica", 10), bg="#242731", fg="#BDC3C7",
                            wraplength=600, justify=tk.LEFT)
        perf_desc.pack(anchor=tk.W, pady=(0, 10), padx=60)
        
        believability_rb = tk.Radiobutton(parent, text="Believability Tournament", 
                                         variable=self.tournament_type, value="believability",
                                         font=("Helvetica", 12),
                                         bg="#242731", fg="#E7E1BD", selectcolor="#242731",
                                         activebackground="#242731", activeforeground="#E7E1BD",
                                         command=self.update_tournament_info)
        believability_rb.pack(anchor=tk.W, pady=5, padx=40)
        
        believe_desc = tk.Label(parent, text="Analyze clue quality, human-likeness, and semantic coherence",
                               font=("Helvetica", 10), bg="#242731", fg="#BDC3C7",
                               wraplength=600, justify=tk.LEFT)
        believe_desc.pack(anchor=tk.W, pady=(0, 10), padx=60)
        
        # Games per matchup
        games_label = tk.Label(parent, text="Games per Matchup:", 
                              font=("Helvetica", 12, "bold"),
                              bg="#242731", fg="#E7E1BD")
        games_label.pack(anchor=tk.W, pady=(20, 5), padx=20)
        
        self.games_var = tk.IntVar(value=2)
        games_scale = tk.Scale(parent, from_=1, to=10, orient=tk.HORIZONTAL,
                              variable=self.games_var, bg="#242731", fg="#E7E1BD",
                              highlightbackground="#242731", troughcolor="#34495E",
                              font=("Helvetica", 10), command=lambda x: self.update_tournament_info())
        games_scale.pack(fill=tk.X, pady=5, padx=40)
        
        # Tournament info
        info_frame = tk.Frame(parent, bg="#242731", bd=1, relief=tk.GROOVE)
        info_frame.pack(fill=tk.X, pady=20, padx=20)
        
        info_label = tk.Label(info_frame, text="Tournament Information:", 
                             font=("Helvetica", 12, "bold"),
                             bg="#242731", fg="#E7E1BD")
        info_label.pack(anchor=tk.W, padx=10, pady=5)
        
        self.info_text = tk.Label(info_frame, text="Configure settings to see tournament details",
                                 font=("Helvetica", 10),
                                 bg="#242731", fg="#BDC3C7", wraplength=600, justify=tk.LEFT)
        self.info_text.pack(anchor=tk.W, padx=10, pady=(5, 10))
    
    def setup_agent_selection_tab(self, parent):
        """Setup agent selection"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg="#242731")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#242731")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Codemaster section
        cm_label = tk.Label(scrollable_frame, text="Codemasters", 
                           font=("Helvetica", 14, "bold"),
                           bg="#242731", fg="#E7E1BD")
        cm_label.pack(anchor=tk.W, pady=(10, 5))
        
        self.codemaster_vars = {}
        codemasters = [
            ("MCTS", "Monte Carlo Tree Search", True),
            ("EMD", "Word Embeddings (Original)", True),
            ("SBERT", "Sentence Transformers", False),
            ("CL", "Curriculum Learning", False),
            ("TOT", "Tree of Thoughts", False)
        ]
        
        for code, name, default in codemasters:
            var = tk.BooleanVar(value=default)
            self.codemaster_vars[code] = var
            
            frame = tk.Frame(scrollable_frame, bg="#242731")
            frame.pack(fill=tk.X, padx=20, pady=2)
            
            cb = tk.Checkbutton(frame, text=f"{name} ({code})", variable=var,
                               font=("Helvetica", 11),
                               bg="#242731", fg="#C13A37", selectcolor="#242731",
                               activebackground="#242731", activeforeground="#C13A37",
                               command=self.update_tournament_info)
            cb.pack(anchor=tk.W)
        
        # Guesser section
        g_label = tk.Label(scrollable_frame, text="Guessers", 
                          font=("Helvetica", 14, "bold"),
                          bg="#242731", fg="#E7E1BD")
        g_label.pack(anchor=tk.W, pady=(20, 5))
        
        self.guesser_vars = {}
        guessers = [
            ("EMD", "Word Embeddings", True),
            ("MCTS", "Monte Carlo Tree Search", True),
            ("SBERT", "Sentence Transformers", False),
            ("Naive", "Simple Embeddings", False)
        ]
        
        for code, name, default in guessers:
            var = tk.BooleanVar(value=default)
            self.guesser_vars[code] = var
            
            frame = tk.Frame(scrollable_frame, bg="#242731")
            frame.pack(fill=tk.X, padx=20, pady=2)
            
            cb = tk.Checkbutton(frame, text=f"{name} ({code})", variable=var,
                               font=("Helvetica", 11),
                               bg="#242731", fg="#4989C5", selectcolor="#242731",
                               activebackground="#242731", activeforeground="#4989C5",
                               command=self.update_tournament_info)
            cb.pack(anchor=tk.W)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_enhanced_settings_tab(self, parent):
        """Setup enhanced tournament settings"""
        # Believability Analysis Settings
        believe_frame = tk.LabelFrame(parent, text="Believability Analysis", 
                                     font=("Helvetica", 12, "bold"),
                                     bg="#242731", fg="#F1C40F")
        believe_frame.pack(fill=tk.X, pady=10, padx=20)
        
        self.believability_enabled = tk.BooleanVar(value=True)
        believe_cb = tk.Checkbutton(believe_frame, text="Enable Believability Analysis", 
                                   variable=self.believability_enabled,
                                   font=("Helvetica", 12),
                                   bg="#242731", fg="#F1C40F", selectcolor="#242731",
                                   activebackground="#242731", activeforeground="#F1C40F")
        believe_cb.pack(anchor=tk.W, padx=10, pady=5)
        
        # Tournament Size Settings
        size_frame = tk.LabelFrame(parent, text="Tournament Size", 
                                  font=("Helvetica", 12, "bold"),
                                  bg="#242731", fg="#E7E1BD")
        size_frame.pack(fill=tk.X, pady=10, padx=20)
        
        self.tournament_size = tk.StringVar(value="medium")
        
        tk.Radiobutton(size_frame, text="Medium (≤200 games)", variable=self.tournament_size, value="medium",
                      bg="#242731", fg="#E7E1BD", selectcolor="#242731").pack(anchor=tk.W, padx=10, pady=2)
        tk.Radiobutton(size_frame, text="Large (≤500 games)", variable=self.tournament_size, value="large",
                      bg="#242731", fg="#E7E1BD", selectcolor="#242731").pack(anchor=tk.W, padx=10, pady=2)
    
    def update_tournament_info(self):
        """Update tournament information display"""
        # Count selected agents
        cm_count = sum(var.get() for var in self.codemaster_vars.values())
        g_count = sum(var.get() for var in self.guesser_vars.values())
        
        if cm_count == 0 or g_count == 0:
            self.info_text.config(text="Please select at least one codemaster and one guesser")
            return
        
        games_per_matchup = self.games_var.get()
        total_teams = cm_count * g_count
        
        # Estimate matchups based on tournament size
        size_limits = {"small": 50, "medium": 200, "large": 500}
        max_games = size_limits.get(self.tournament_size.get(), 200)
        max_matchups = max_games // games_per_matchup
        
        total_matchups = min(total_teams * (total_teams - 1), max_matchups)
        total_games = total_matchups * games_per_matchup
        
        tournament_type = self.tournament_type.get()
        analysis_overhead = 1.5 if tournament_type == "believability" else 1.0
        estimated_time = total_games * 0.5 * analysis_overhead / 60
        
        info_text = f"""Tournament Details:
• Type: {tournament_type.title()} Tournament
• {cm_count} codemasters, {g_count} guessers selected
• {total_teams} total teams
• {total_matchups} unique matchups
• {games_per_matchup} games per matchup
• {total_games} total games
• Estimated time: {estimated_time:.1f} minutes"""
        
        if tournament_type == "believability":
            info_text += "\n• Includes believability analysis"
        
        self.info_text.config(text=info_text)
    
    def start_tournament(self):
        """Start the enhanced tournament"""
        # Validate selection
        selected_cm = [code for code, var in self.codemaster_vars.items() if var.get()]
        selected_g = [code for code, var in self.guesser_vars.items() if var.get()]
        
        if not selected_cm:
            messagebox.showerror("Error", "Please select at least one codemaster")
            return
        
        if not selected_g:
            messagebox.showerror("Error", "Please select at least one guesser")
            return
        
        # Create result configuration
        size_limits = {"small": 50, "medium": 200, "large": 500}
        max_games = size_limits.get(self.tournament_size.get(), 200)
        
        self.result = {
            'tournament_type': self.tournament_type.get(),
            'codemasters': selected_cm,
            'guessers': selected_g,
            'games_per_matchup': self.games_var.get(),
            'believability_analysis': (self.tournament_type.get() == "believability" and 
                                     self.believability_enabled.get()),
            'max_games': max_games
        }
        
        self.window.destroy()
    
    def cancel(self):
        """Cancel tournament setup"""
        self.result = None
        self.window.destroy()

# ==================== ENHANCED PROGRESS WINDOW ====================

class EnhancedTournamentProgressWindow:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.cancelled = False
        self.start_time = time.time()
        
        # Create window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Enhanced Tournament in Progress")
        self.window.geometry("700x500")
        self.window.configure(bg="#242731")
        self.window.transient(parent.root)
        self.window.grab_set()
        
        # Center the window
        self.window.geometry("+{}+{}".format(
            parent.root.winfo_rootx() + 100,
            parent.root.winfo_rooty() + 100
        ))
        
        # Prevent window closing
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = tk.Frame(self.window, bg="#242731")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_text = f"Enhanced {self.config.get('tournament_type', 'Performance').title()} Tournament"
        title_label = tk.Label(main_frame, text=title_text, 
                              font=("Helvetica", 18, "bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Progress bars frame
        progress_frame = tk.Frame(main_frame, bg="#242731")
        progress_frame.pack(fill=tk.X, pady=10)
        
        # Game progress
        tk.Label(progress_frame, text="Game Progress:", font=("Helvetica", 12, "bold"),
                bg="#242731", fg="#E7E1BD").pack(anchor=tk.W)
        
        self.game_progress_var = tk.DoubleVar()
        self.game_progress_bar = ttk.Progressbar(progress_frame, variable=self.game_progress_var,
                                               maximum=100, length=600)
        self.game_progress_bar.pack(pady=5)
        
        self.game_progress_label = tk.Label(progress_frame, text="Preparing tournament...",
                                          font=("Helvetica", 11),
                                          bg="#242731", fg="#E7E1BD")
        self.game_progress_label.pack(pady=5)
        
        # Analysis progress (for believability tournaments)
        if self.config.get('believability_analysis', False):
            tk.Label(progress_frame, text="Analysis Progress:", font=("Helvetica", 12, "bold"),
                    bg="#242731", fg="#F1C40F").pack(anchor=tk.W, pady=(10, 0))
            
            self.analysis_progress_var = tk.DoubleVar()
            self.analysis_progress_bar = ttk.Progressbar(progress_frame, variable=self.analysis_progress_var,
                                                       maximum=100, length=600)
            self.analysis_progress_bar.pack(pady=5)
            
            self.analysis_progress_label = tk.Label(progress_frame, text="Waiting for games to complete...",
                                                  font=("Helvetica", 11),
                                                  bg="#242731", fg="#F1C40F")
            self.analysis_progress_label.pack(pady=5)
        
        # Current match info
        self.match_label = tk.Label(main_frame, text="",
                                   font=("Helvetica", 11),
                                   bg="#242731", fg="#BDC3C7", wraplength=600)
        self.match_label.pack(pady=10)
        
        # Stats frame
        stats_frame = tk.Frame(main_frame, bg="#242731", bd=1, relief=tk.GROOVE)
        stats_frame.pack(pady=20, fill=tk.X)
        
        self.stats_label = tk.Label(stats_frame, 
                                   text="Games completed: 0\nTime elapsed: 0:00\nEstimated remaining: --",
                                   font=("Helvetica", 11),
                                   bg="#242731", fg="#BDC3C7", justify=tk.LEFT)
        self.stats_label.pack(padx=10, pady=10)
        
        # Log area
        log_frame = tk.Frame(main_frame, bg="#242731")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        log_label = tk.Label(log_frame, text="Tournament Log:", 
                            font=("Helvetica", 10, "bold"),
                            bg="#242731", fg="#E7E1BD")
        log_label.pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                font=("Consolas", 9),
                                                bg="#1E1F29", fg="#E7E1BD", 
                                                relief=tk.FLAT, bd=1, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Cancel button
        self.cancel_button = tk.Button(main_frame, text="Cancel Tournament", 
                                      command=self.cancel_tournament,
                                      font=("Helvetica", 12),
                                      bg="#F44336", fg="white", relief=tk.FLAT, bd=0, 
                                      padx=20, pady=8)
        self.cancel_button.pack(pady=10)
    
    def update_enhanced_progress(self, completed, total, current_match="", analysis_stage=""):
        """Update enhanced progress with analysis information"""
        if not self.cancelled:
            # Update game progress
            progress = (completed / total) * 100 if total > 0 else 0
            self.game_progress_var.set(progress)
            self.game_progress_label.config(text=f"Games: {completed}/{total} ({progress:.1f}%)")
            
            if current_match:
                self.match_label.config(text=f"Current: {current_match}")
            
            # Update analysis progress if applicable
            if hasattr(self, 'analysis_progress_label') and analysis_stage:
                self.analysis_progress_label.config(text=f"Analysis: {analysis_stage}")
            
            # Update stats
            elapsed = time.time() - self.start_time
            if completed > 0:
                avg_time = elapsed / completed
                remaining_games = total - completed
                estimated_remaining = remaining_games * avg_time
                
                stats_text = f"""Games completed: {completed}
Time elapsed: {self.format_time(elapsed)}
Estimated remaining: {self.format_time(estimated_remaining)}
Avg game time: {avg_time:.1f}s"""
            else:
                stats_text = f"""Games completed: {completed}
Time elapsed: {self.format_time(elapsed)}
Estimated remaining: --"""
            
            self.stats_label.config(text=stats_text)
    
    def add_analysis_log(self, message, stage="Info"):
        """Add analysis-specific log entry"""
        if not self.cancelled:
            timestamp = time.strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {stage}: {message}"
            
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{formatted_message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
    
    def format_time(self, seconds):
        """Format time duration"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def cancel_tournament(self):
        """Cancel the tournament"""
        result = messagebox.askyesno("Cancel Tournament", 
                                    "Are you sure you want to cancel the enhanced tournament?\nProgress will be lost.")
        if result:
            self.cancelled = True
            self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)
            self.add_analysis_log("Tournament cancellation requested...", "System")
    
    def on_close(self):
        """Prevent closing window directly"""
        pass
    
    def close(self):
        """Close the progress window"""
        self.window.destroy()

# ==================== ENHANCED RESULTS WINDOW ====================

class EnhancedTournamentResultsWindow:
    def __init__(self, parent, results):
        self.parent = parent
        self.results = results
        
        # Create window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Enhanced Tournament Results")
        self.window.geometry("1400x900")
        self.window.configure(bg="#242731")
        self.window.transient(parent.root)
        
        # Center the window
        self.window.geometry("+{}+{}".format(
            parent.root.winfo_rootx() + 50,
            parent.root.winfo_rooty() + 50
        ))
        
        self.create_widgets()
    
    def create_widgets(self):
        main_frame = tk.Frame(self.window, bg="#242731")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title with tournament type
        tournament_type = getattr(self.results, 'tournament_name', 'Enhanced Tournament')
        title_label = tk.Label(main_frame, text=f"Enhanced Tournament Results: {tournament_type}", 
                              font=("Helvetica", 18, "bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Summary stats
        summary_frame = tk.Frame(main_frame, bg="#242731", bd=1, relief=tk.GROOVE)
        summary_frame.pack(fill=tk.X, pady=(0, 20))
        
        total_games = getattr(self.results, 'total_games', 0)
        summary_text = f"Total Games: {total_games}"
        
        # Add believability info if available
        if hasattr(self.results, 'believability_analysis'):
            analysis = self.results.believability_analysis
            if 'enhanced_metrics_count' in analysis:
                summary_text += f" | Clues Analyzed: {analysis['enhanced_metrics_count']}"
        
        summary_label = tk.Label(summary_frame, text=summary_text,
                                font=("Helvetica", 12), bg="#242731", fg="#E7E1BD")
        summary_label.pack(pady=10)
        
        # Create notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Team Rankings tab
        team_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(team_frame, text="Team Rankings")
        self.setup_enhanced_team_rankings(team_frame)
        
        # Agent Rankings tab
        agent_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(agent_frame, text="Agent Rankings")
        self.setup_enhanced_agent_rankings(agent_frame)
        
        # Believability tab (if available)
        if hasattr(self.results, 'believability_analysis') and self.results.believability_analysis:
            believe_frame = tk.Frame(notebook, bg="#242731")
            notebook.add(believe_frame, text="Believability Analysis")
            self.setup_enhanced_believability_tab(believe_frame)
        
        # Raw Data tab
        raw_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(raw_frame, text="Raw Data")
        self.setup_raw_data_tab(raw_frame)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#242731")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Export button
        export_btn = tk.Button(button_frame, text="Export Enhanced Results", 
                              command=self.export_enhanced_results,
                              font=("Helvetica", 12),
                              bg="#FF9800", fg="white", relief=tk.FLAT, bd=0, 
                              padx=20, pady=8)
        export_btn.pack(side=tk.LEFT)
        
        # Close button
        close_btn = tk.Button(button_frame, text="Close", command=self.window.destroy,
                             font=("Helvetica", 12),
                             bg="#9E9E9E", fg="white", relief=tk.FLAT, bd=0, 
                             padx=20, pady=8)
        close_btn.pack(side=tk.RIGHT)
    
    def setup_enhanced_team_rankings(self, parent):
        """Setup enhanced team rankings"""
        # Create treeview with enhanced columns
        columns = ("Rank", "Team", "Win Rate", "W-L", "Games", "TrueSkill", "Conservative")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("Rank", width=60, anchor=tk.CENTER)
        tree.column("Team", width=250, anchor=tk.W)
        tree.column("Win Rate", width=80, anchor=tk.CENTER)
        tree.column("W-L", width=80, anchor=tk.CENTER)
        tree.column("Games", width=60, anchor=tk.CENTER)
        tree.column("TrueSkill", width=100, anchor=tk.CENTER)
        tree.column("Conservative", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with team data
        if hasattr(self.results, 'team_rankings') and self.results.team_rankings:
            for i, (team_key, stats) in enumerate(self.results.team_rankings, 1):
                win_rate = getattr(stats, 'win_rate', 0) if hasattr(stats, 'win_rate') else stats.wins / max(1, stats.total_games)
                wins = getattr(stats, 'wins', 0)
                losses = getattr(stats, 'losses', 0)
                total_games = getattr(stats, 'total_games', 0)
                trueskill = getattr(stats, 'trueskill_rating', None)
                conservative = getattr(stats, 'conservative_skill', 0) if hasattr(stats, 'conservative_skill') else 0
                
                trueskill_text = f"{trueskill.mu:.1f}" if trueskill else "N/A"
                
                tree.insert("", tk.END, values=(
                    i,
                    team_key,
                    f"{win_rate:.1%}",
                    f"{wins}-{losses}",
                    total_games,
                    trueskill_text,
                    f"{conservative:.2f}"
                ))
        else:
            tree.insert("", tk.END, values=("", "No team data available", "", "", "", "", ""))
    
    def setup_enhanced_agent_rankings(self, parent):
        """Setup enhanced agent rankings"""
        # Create treeview for agent rankings
        columns = ("Rank", "Agent", "Type", "Win Rate", "W-L", "Games", "Performance")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("Rank", width=50, anchor=tk.CENTER)
        tree.column("Agent", width=150, anchor=tk.W)
        tree.column("Type", width=100, anchor=tk.CENTER)
        tree.column("Win Rate", width=80, anchor=tk.CENTER)
        tree.column("W-L", width=80, anchor=tk.CENTER)
        tree.column("Games", width=60, anchor=tk.CENTER)
        tree.column("Performance", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with agent data
        if hasattr(self.results, 'agent_rankings') and self.results.agent_rankings:
            for i, (agent_name, metrics) in enumerate(self.results.agent_rankings, 1):
                agent_type = getattr(metrics, 'agent_type', 'Unknown')
                wins = getattr(metrics, 'wins', 0)
                losses = getattr(metrics, 'losses', 0)
                games_played = getattr(metrics, 'games_played', 0)
                win_rate = wins / max(1, games_played)
                
                # Get performance metric
                performance = getattr(metrics, 'role_based_rating', None)
                perf_text = f"{performance.mu:.1f}" if performance else "N/A"
                
                tree.insert("", tk.END, values=(
                    i,
                    agent_name,
                    agent_type.title(),
                    f"{win_rate:.1%}",
                    f"{wins}-{losses}",
                    games_played,
                    perf_text
                ))
        else:
            tree.insert("", tk.END, values=("", "No agent data available", "", "", "", "", ""))
    
    def setup_enhanced_believability_tab(self, parent):
        """Setup believability analysis tab"""
        # Create label
        believe_label = tk.Label(parent, text="Believability Analysis", 
                                font=("Helvetica", 14, "bold"),
                                bg="#242731", fg="#F1C40F")
        believe_label.pack(pady=(0, 10))
        
        # Check if we have believability data
        analysis = self.results.believability_analysis
        if not analysis or 'top_believable_codemasters' not in analysis:
            no_data_label = tk.Label(parent, 
                                    text="No believability data available.\nEnable believability analysis in tournament settings.",
                                    font=("Helvetica", 12),
                                    bg="#242731", fg="#BDC3C7")
            no_data_label.pack(expand=True)
            return
        
        # Create treeview for believability
        b_columns = ("Rank", "Agent", "Believability", "Details")
        b_tree = ttk.Treeview(parent, columns=b_columns, show="headings", height=15)
        
        for col in b_columns:
            b_tree.heading(col, text=col)
            b_tree.column(col, width=150, anchor=tk.CENTER)
        
        # Add scrollbar
        b_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=b_tree.yview)
        b_tree.configure(yscrollcommand=b_scrollbar.set)
        
        # Pack widgets
        b_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        b_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with believability data
        top_believable = analysis.get('top_believable_codemasters', [])
        for i, (name, score) in enumerate(top_believable, 1):
            b_tree.insert("", tk.END, values=(
                i,
                name,
                f"{score:.3f}",
                "Codemaster Analysis"
            ))
    
    def setup_raw_data_tab(self, parent):
        """Create scrollable text area for raw data"""
        text_frame = tk.Frame(parent, bg="#242731")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        raw_text = scrolledtext.ScrolledText(text_frame, 
                                           font=("Consolas", 9),
                                           bg="#1E1F29", fg="#E7E1BD", 
                                           relief=tk.FLAT, bd=1, 
                                           wrap=tk.WORD)
        raw_text.pack(fill=tk.BOTH, expand=True)
        
        # Add raw data
        try:
            if hasattr(self.results, 'to_dict'):
                raw_data = json.dumps(self.results.to_dict(), indent=2, default=str)
            else:
                raw_data = str(self.results)
            raw_text.insert(tk.END, raw_data)
        except Exception as e:
            raw_text.insert(tk.END, f"Error displaying raw data: {e}")
        
        raw_text.config(state=tk.DISABLED)
    
    def export_enhanced_results(self):
        """Export enhanced tournament results"""
        from tkinter import filedialog
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Enhanced Tournament Results"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    # Export JSON
                    if hasattr(self.results, 'to_dict'):
                        data = self.results.to_dict()
                    else:
                        data = {"message": "Enhanced results format not available"}
                    
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                
                elif filename.endswith('.csv'):
                    # Export CSV
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Export team rankings
                        writer.writerow(["Enhanced Tournament Results"])
                        writer.writerow(["Team Rankings"])
                        writer.writerow(["Rank", "Team", "Win Rate", "Wins", "Losses", "Games", "TrueSkill"])
                        
                        if hasattr(self.results, 'team_rankings'):
                            for i, (team_key, stats) in enumerate(self.results.team_rankings, 1):
                                win_rate = getattr(stats, 'win_rate', 0) if hasattr(stats, 'win_rate') else stats.wins / max(1, stats.total_games)
                                wins = getattr(stats, 'wins', 0)
                                losses = getattr(stats, 'losses', 0)
                                total_games = getattr(stats, 'total_games', 0)
                                trueskill = getattr(stats, 'trueskill_rating', None)
                                trueskill_text = f"{trueskill.mu:.2f}" if trueskill else "N/A"
                                
                                writer.writerow([i, team_key, f"{win_rate:.3f}", wins, losses, total_games, trueskill_text])
                
                messagebox.showinfo("Export Complete", f"Enhanced results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

# ==================== MAIN ====================

if __name__ == "__main__":
    root = tk.Tk()
    app = CodenamesGUI(root)
    root.mainloop().Radiobutton(size_frame, text="Small (≤50 games)", variable=self.tournament_size, value="small",
                      bg="#242731", fg="#E7E1BD", selectcolor="#242731").pack(anchor=tk.W, padx=10, pady=2)
                        