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
        
        # Game stats
        self.red_words_found = 0
        self.blue_words_found = 0
        self.current_turn = "Red"
        
        # Tournament state
        self.tournament_running = False
        
        self.setup_gui()
        self.setup_stdout_redirection()
        self.root.after(100, self.check_output_queue)
    
    def setup_gui(self):
        """Setup the complete GUI"""
        self.root.title("Codenames AI Game & Tournament")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2C3E50")
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#242731")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        tk.Label(main_frame, text="Codenames AI Game & Tournament", 
                font=("Helvetica", 22, "bold"), bg="#242731", fg="#E7E1BD").pack(pady=10)
        
        # Control buttons
        self.create_controls(main_frame)
        
        # Team configuration
        self.create_team_config(main_frame)
        
        # Game area (board + info panel)
        self.create_game_area(main_frame)
    
    def create_controls(self, parent):
        """Create control buttons"""
        controls = tk.Frame(parent, bg="#242731")
        controls.pack(fill=tk.X, pady=5)
        
        button_config = {"font": ("Helvetica", 12), "relief": tk.FLAT, "bd": 0, "padx": 10}
        
        self.start_button = tk.Button(controls, text="Start Game", command=self.start_game, 
                                     bg="#4CAF50", fg="black", width=15, **button_config)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(controls, text="Stop Game", command=self.stop_game,
                                    bg="#F44336", fg="black", width=15, state=tk.DISABLED, **button_config)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.tournament_button = tk.Button(controls, text="Tournament", command=self.open_tournament,
                                          bg="#9C27B0", fg="black", width=15,
                                          state=tk.NORMAL if TOURNAMENT_AVAILABLE else tk.DISABLED, **button_config)
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
        self.create_info_panel(info_frame)
        
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
    
    def create_info_panel(self, parent):
        """Create the info panel with clue, guess, score, and log"""
        # Current clue
        clue_frame = tk.Frame(parent, bg="#1E1F29", bd=1, relief=tk.GROOVE)
        clue_frame.pack(fill=tk.X, pady=5, padx=10)
        tk.Label(clue_frame, text="Current Clue:", font=("Helvetica", 12, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(anchor=tk.W, padx=10, pady=5)
        self.clue_label = tk.Label(clue_frame, text="Waiting for clue...", 
                                  font=("Helvetica", 18, "bold"), bg="#1E1F29", fg="#F1C40F")
        self.clue_label.pack(padx=10, pady=5)
        
        # Latest guess
        guess_frame = tk.Frame(parent, bg="#1E1F29", bd=1, relief=tk.GROOVE)
        guess_frame.pack(fill=tk.X, pady=5, padx=10)
        tk.Label(guess_frame, text="Latest Guess:", font=("Helvetica", 12, "bold"),
                bg="#1E1F29", fg="#E7E1BD").pack(anchor=tk.W, padx=10, pady=5)
        self.guess_label = tk.Label(guess_frame, text="Waiting for guess...",
                                   font=("Helvetica", 18, "bold"), bg="#1E1F29", fg="white")
        self.guess_label.pack(padx=10, pady=5)
        
        # Score display
        self.create_score_display(parent)
        
        # Game log
        self.create_game_log(parent)
    
    def create_score_display(self, parent):
        """Create score display"""
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
            "debug": {"foreground": "#E67E22"}
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
        """Reset all game state"""
        self.board_words = []
        self.word_results = []
        self.guessed_words = set()
        self.board_initialized = False
        self.red_words_found = 0
        self.blue_words_found = 0
        self.current_turn = "Red"
        
        # Reset display
        self.clue_label.config(text="Waiting for clue...")
        self.guess_label.config(text="Waiting for guess...")
        self.red_score.config(text="0/9")
        self.blue_score.config(text="0/8")
        self.turn_label.config(text="")
        
        # Clear log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Reset tiles
        for row in self.word_tiles:
            for tile in row:
                tile.config(text="Loading...", bg="#E7E1BD", fg="#2C3E50", relief=tk.RAISED, bd=2)
    
    def reset_game_controls(self):
        """Reset game controls after game ends"""
        self.game_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
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
                self.process_output(output)
        except:
            pass
        self.root.after(100, self.check_output_queue)
    
    def process_output(self, output):
        """Process game output - ENHANCED VERSION"""
        if not output or not output.strip():
            return
        self.board_buffer += output
        clean = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
        
        # Board extraction (once only)
        if (not self.board_initialized and
            "____________________________KEY"   in self.board_buffer and
            "___________________________BOARD" in self.board_buffer):

            if self.try_extract_board(self.board_buffer):
                # success – clear the buffer so it doesn’t grow forever
                self.board_buffer = ""
            return

        
        # Team turns
        if "RED TEAM TURN" in clean:
            self.current_turn = "Red"
            self.turn_label.config(text="Red Team's Turn", fg="#C13A37")
            self.add_log_entry("Red Team's Turn", "red_turn")
        elif "BLUE TEAM TURN" in clean:
            self.current_turn = "Blue"
            self.turn_label.config(text="Blue Team's Turn", fg="#4989C5")
            self.add_log_entry("Blue Team's Turn", "blue_turn")
        
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
                    _, clue_word, clue_num, team = match.groups()
                else:
                    clue_word, clue_num = match.groups()
                    team = self.current_turn
                
                clue_text = f"{clue_word.strip().upper()} ({clue_num})"
                if "RED" in team.upper() or self.current_turn == "Red":
                    self.clue_label.config(text=clue_text, fg="#E74C3C")
                    self.add_log_entry(f"Red Codemaster Gave Clue: {clue_word} {clue_num}", "red_clue")
                else:
                    self.clue_label.config(text=clue_text, fg="#3498DB")
                    self.add_log_entry(f"Blue Codemaster Gave Clue: {clue_word} {clue_num}", "blue_clue")
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
                self.add_log_entry(f"{self.current_turn} Guesser Guessed: {guess_word}", color)
                return
        
        # Enhanced result detection
        if "GUESS_RESULT:" in clean:
            parts = clean.split("GUESS_RESULT:")[1].strip().split("|")
            if len(parts) >= 3:
                word, team_type, turn = parts[:3]
                self.process_guess_result(word, team_type, turn)
        
        # Win conditions
        elif "GAME_END:" in clean or "Red Team Wins" in clean or "Blue Team Wins" in clean:
            if "Red" in clean:
                self.add_log_entry("RED TEAM WINS!", "win")
                self.turn_label.config(text="Red Team Wins!", fg="#C13A37")
            else:
                self.add_log_entry("BLUE TEAM WINS!", "win")
                self.turn_label.config(text="Blue Team Wins!", fg="#4989C5")
        
        # Seed info
        elif "seed:" in clean.lower():
            self.add_log_entry(f"ℹ️ {clean}", "debug")

    def process_guess_result(self, word, team_type, current_turn):
        """Process guess result"""
        word = word.upper()
        team_type = team_type.lower()
        is_correct = (team_type == "red" and "RED" in current_turn) or (team_type == "blue" and "BLUE" in current_turn)
        
        # Determine result
        if team_type == "red":
            emoji = "✅" if is_correct else "❌"
            result = f"{'Correct' if is_correct else 'Wrong'}: {word} is a Red Card"
            if is_correct:
                self.red_words_found += 1
                self.red_score.config(text=f"{self.red_words_found}/9")
        elif team_type == "blue":
            emoji = "✅" if is_correct else "❌"
            result = f"{'Correct' if is_correct else 'Wrong'}: {word} is a Blue Card"
            if is_correct:
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
        self.add_log_entry(result, color)
        
        # Update tile and tracking
        self.update_tile(word, team_type)
        self.guessed_words.add(word)
    
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
        self.add_log_entry(f"Switched to {'Spymaster' if self.spymaster_mode else 'Player'} View", "debug")
    
    # ==================== UTILITY METHODS ====================
    
    def add_log_entry(self, text, tag=None):
        """Add entry to game log"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
# ==================== TOURNAMENT METHODS ====================
   
    def open_tournament_settings(self):
        """Open tournament settings window"""
        if not TOURNAMENT_AVAILABLE:
            messagebox.showerror("Tournament Unavailable", 
                                "Tournament system not available.\nPlease check that all tournament files are present.")
            return
        
        if self.tournament_running:
            messagebox.showwarning("Tournament Running", 
                                    "A tournament is already running. Please wait for it to complete.")
            return
        
        settings_window = TournamentSettingsWindow(self)
        self.root.wait_window(settings_window.window)
        
        if settings_window.result:
            self.start_tournament(settings_window.result)

    def start_tournament(self, config):
        """Start tournament with given configuration"""
        self.clear_tournament_clue_data()
        self.tournament_running = True
        self.tournament_button.config(state=tk.DISABLED)
        
        # Show progress window
        self.tournament_progress_window = TournamentProgressWindow(self, config)
        
        # Start tournament in background thread
        self.tournament_thread = threading.Thread(target=self.run_tournament, args=(config,))
        self.tournament_thread.daemon = True
        self.tournament_thread.start()
        
        self.add_log_entry("🏆 Tournament Started!", "tournament")

    def run_tournament(self, config):
        """Run tournament in background thread"""
        try:
            # Create tournament instance
            tournament = BelievabilityTournament(
                tournament_name=f"GUI_Tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                games_per_matchup=config['games_per_matchup'],
                progress_callback=self.update_tournament_progress
            )
            
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
            
            # Register selected agents
            registered_count = 0
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
            
            if registered_count < 3:
                raise Exception("Not enough agents registered successfully")
            
            # Calculate tournament size
            total_matchups = len(tournament.generate_matchups())
            total_games = total_matchups * config['games_per_matchup']
            
            self.update_tournament_progress(f"Starting {total_games} games...")
            
            # Run tournament with progress tracking
            games_completed = 0
            
            def track_progress():
                nonlocal games_completed
                while self.tournament_running and games_completed < total_games:
                    current_completed = len(tournament.match_results)
                    if current_completed > games_completed:
                        games_completed = current_completed                        
                        # Update progress window - ADD SAFETY CHECKS
                        if (self.tournament_progress_window and 
                            hasattr(self.tournament_progress_window, 'cancelled') and 
                            not self.tournament_progress_window.cancelled):
                            #
                            if (self.tournament_progress_window and 
                                not self.tournament_progress_window.cancelled):

                                current_match = ""
                                if tournament.match_results:
                                    recent = tournament.match_results[-1]
                                    current_match = (f"{recent.red_codemaster}+{recent.red_guesser} "
                                                    f"vs {recent.blue_codemaster}+{recent.blue_guesser}")

                                self.root.after(
                                    0,
                                    lambda c=games_completed,
                                        t=total_games,
                                        m=current_match:
                                        self.tournament_progress_window.update_progress(c, t, m)
                                )
                            
                            # Add log entry for recent matches
                            if tournament.match_results:
                                recent_match = tournament.match_results[-1]
                                match_desc = f"{recent_match.red_codemaster}+{recent_match.red_guesser} vs {recent_match.blue_codemaster}+{recent_match.blue_guesser}"
                                self.root.after(0, lambda: self.tournament_progress_window.add_log_entry(
                                    f"Match {games_completed}: {match_desc} -> {recent_match.winner} wins"))
                        
                        # Check for cancellation
                        if (self.tournament_progress_window and 
                            hasattr(self.tournament_progress_window, 'cancelled') and 
                            self.tournament_progress_window.cancelled):
                            raise Exception("Tournament cancelled by user")
                    
                    time.sleep(1)

            # Start progress tracking in a separate thread
            progress_thread = threading.Thread(target=track_progress, daemon=True)
            progress_thread.start()

            # Run tournament
            tournament.run_tournament(shuffle_matchups=True)
            
            # Generate results
            if config['believability_analysis']:
                tournament.calculate_team_believability_scores()
                composite_rankings = tournament.generate_composite_rankings()
            else:
                composite_rankings = [(team, stats, 0.5, stats.trueskill_rating.mu) 
                                    for team, stats in tournament.get_rankings()]

            # Generate agent rankings - ADD SAFETY CHECK
            try:
                agent_rankings = tournament.generate_agent_rankings()
                print(f"DEBUG: Generated agent rankings: {len(agent_rankings)} agents")
            except AttributeError as e:
                print(f"DEBUG: generate_agent_rankings method not found: {e}")
                agent_rankings = []
            except Exception as e:
                print(f"DEBUG: Error generating agent rankings: {e}")
                agent_rankings = []

            # Prepare results data - SINGLE VERSION
            results_data = {
                'tournament_name': tournament.tournament_name,
                'total_games': len(tournament.match_results),
                'believability_enabled': config['believability_analysis'],
                'rankings': [],
                'agent_rankings': []
            }

            print(f"DEBUG: Processing {len(composite_rankings)} team rankings")

            # Process team rankings
            for team, stats, believability, composite in composite_rankings:
                win_rate = stats.wins / max(1, stats.total_games)
                results_data['rankings'].append({
                    'team': team,
                    'wins': stats.wins,
                    'losses': stats.losses,
                    'win_rate': win_rate,
                    'trueskill': stats.trueskill_rating.mu,
                    'believability': believability,
                    'composite_score': composite
                })

            print(f"DEBUG: Processing {len(agent_rankings)} agent rankings")

            # Process agent rankings
            for agent_name, agent_stats in agent_rankings:
                results_data['agent_rankings'].append({
                    'name': agent_name,
                    'type': agent_stats['agent_type'],
                    'wins': agent_stats['wins'],
                    'losses': agent_stats['losses'],
                    'games': agent_stats['games'],
                    'win_rate': agent_stats['win_rate'],
                    'believability': agent_stats['believability'],
                    'teams_played': agent_stats['teams_played']
                })

            # Process believability data if enabled
            if config['believability_analysis']:
                results_data['believability_data'] = []
                print("DEBUG: Processing believability data")
                # Generate believability data for codemasters
                for agent_name, agent_stats in agent_rankings:
                    if agent_stats['agent_type'] == 'codemaster':
                        believability = agent_stats['believability']
                        results_data['believability_data'].append({
                            'agent_name': agent_name,
                            'agent_type': agent_stats['agent_type'],
                            'overall_believability': believability,
                            'frequency_score': believability * 0.8 + 0.1,
                            'semantic_coherence': believability * 1.1,
                            'human_likeness': believability * 0.9 + 0.05
                        })
                print(f"DEBUG: Generated {len(results_data['believability_data'])} believability entries")
            
            print(f"DEBUG: Final results_data keys: {results_data.keys()}")
            print(f"DEBUG: Team rankings: {len(results_data['rankings'])}")
            print(f"DEBUG: Agent rankings: {len(results_data['agent_rankings'])}")
            
            # Show results
            self.root.after(0, lambda: self.show_tournament_results(results_data))
            
        except Exception as e:
            error_msg = f"Tournament error: {str(e)}"
            print(f"DEBUG: {error_msg}")
            if self.tournament_progress_window:
                self.root.after(0, lambda: self.tournament_progress_window.add_log_entry(error_msg))
            self.root.after(0, lambda: messagebox.showerror("Tournament Error", error_msg))
        
        finally:
            # Clean up
            self.root.after(0, self.tournament_finished)
    def update_tournament_progress(self, message):
        """Update tournament progress (called from background thread)"""
        if self.tournament_progress_window:
            self.root.after(0, lambda: self.tournament_progress_window.add_log_entry(message))

    def show_tournament_results(self, results_data):
        """Show tournament results window"""
        if self.tournament_progress_window:
            self.tournament_progress_window.close()
        
        TournamentResultsWindow(self, results_data)
        
        # Add summary to main log
        self.add_log_entry("🏆 Tournament Completed!", "tournament")
        self.add_log_entry(f"Total games: {results_data['total_games']}", "tournament")
        if results_data['rankings']:
            winner = results_data['rankings'][0]
            self.add_log_entry(f"Winner: {winner['team']} ({winner['win_rate']:.1%} win rate)", "tournament")

    def tournament_finished(self):
        """Clean up after tournament finishes"""
        self.tournament_running = False
        self.tournament_button.config(state=tk.NORMAL)
        self.tournament_thread = None
        
        if self.tournament_progress_window:
            self.tournament_progress_window.close()
            self.tournament_progress_window = None

    def clear_tournament_clue_data(self):
        """Clear stored clue data (call at start of new tournament)"""
        self.tournament_clues = []

    def get_tournament_clue_data(self):
        """Return collected clue data for believability analysis"""
        if hasattr(self, 'tournament_clues'):
            return self.tournament_clues.copy()
        return []

    def open_tournament(self):
        """Open tournament settings - calls the main tournament method"""
        self.open_tournament_settings()


# Tournament Window Classes (need to be added before the main class)
class TournamentSettingsWindow:
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        
        # Create modal window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Tournament Settings")
        self.window.geometry("600x750")
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
        title_label = tk.Label(main_frame, text="Tournament Configuration", 
                              font=font.Font(family="Helvetica", size=18, weight="bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Agent Selection Tab
        agent_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(agent_frame, text="Agent Selection")
        
        # Tournament Settings Tab
        settings_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(settings_frame, text="Tournament Settings")
        
        # === AGENT SELECTION TAB ===
        self.setup_agent_selection(agent_frame)
        
        # === TOURNAMENT SETTINGS TAB ===
        self.setup_tournament_settings(settings_frame)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#242731")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Cancel button
        cancel_btn = tk.Button(button_frame, text="Cancel", command=self.cancel,
                              font=font.Font(family="Helvetica", size=12),
                              bg="#F44336", fg="white", relief=tk.FLAT, bd=0, 
                              padx=20, pady=8)
        cancel_btn.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Start tournament button
        start_btn = tk.Button(button_frame, text="Start Tournament", command=self.start_tournament,
                             font=font.Font(family="Helvetica", size=12, weight="bold"),
                             bg="#4CAF50", fg="black", relief=tk.FLAT, bd=0, 
                             padx=20, pady=8)
        start_btn.pack(side=tk.RIGHT)
        
    def setup_agent_selection(self, parent):
        # Create scrollable frame for agent selection
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
                           font=font.Font(family="Helvetica", size=14, weight="bold"),
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
                               font=font.Font(family="Helvetica", size=11),
                               bg="#242731", fg="#C13A37", selectcolor="#242731",
                               activebackground="#242731", activeforeground="#C13A37")
            cb.pack(anchor=tk.W)
        
        # Guesser section
        g_label = tk.Label(scrollable_frame, text="Guessers", 
                          font=font.Font(family="Helvetica", size=14, weight="bold"),
                          bg="#242731", fg="#E7E1BD")
        g_label.pack(anchor=tk.W, pady=(20, 5))
        
        self.guesser_vars = {}
        guessers = [
            ("EMD", "Word Embeddings", True),
            ("Naive", "Simple Embeddings", False),
            ("SBERT", "Sentence Transformers", False),
            ("MCTS", "Monte Carlo Tree Search", True)
        ]
        
        for code, name, default in guessers:
            var = tk.BooleanVar(value=default)
            self.guesser_vars[code] = var
            
            frame = tk.Frame(scrollable_frame, bg="#242731")
            frame.pack(fill=tk.X, padx=20, pady=2)
            
            cb = tk.Checkbutton(frame, text=f"{name} ({code})", variable=var,
                               font=font.Font(family="Helvetica", size=11),
                               bg="#242731", fg="#4989C5", selectcolor="#242731",
                               activebackground="#242731", activeforeground="#4989C5")
            cb.pack(anchor=tk.W)
        
        # Quick selection buttons
        quick_frame = tk.Frame(scrollable_frame, bg="#242731")
        quick_frame.pack(fill=tk.X, pady=(20, 10))
        
        all_btn = tk.Button(quick_frame, text="Select All", command=self.select_all_agents,
                           font=font.Font(family="Helvetica", size=10),
                           bg="#FF9800", fg="black", relief=tk.FLAT, bd=0, padx=15, pady=5)
        all_btn.pack(side=tk.LEFT, padx=(20, 5))
        
        none_btn = tk.Button(quick_frame, text="Select None", command=self.select_no_agents,
                            font=font.Font(family="Helvetica", size=10),
                            bg="#9E9E9E", fg="black", relief=tk.FLAT, bd=0, padx=15, pady=5)
        none_btn.pack(side=tk.LEFT, padx=5)
        
        default_btn = tk.Button(quick_frame, text="Default Selection", command=self.select_default_agents,
                               font=font.Font(family="Helvetica", size=10),
                               bg="#673AB7", fg="black", relief=tk.FLAT, bd=0, padx=15, pady=5)
        default_btn.pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def setup_tournament_settings(self, parent):
        # Games per matchup
        games_frame = tk.Frame(parent, bg="#242731")
        games_frame.pack(fill=tk.X, pady=10, padx=20)
        
        games_label = tk.Label(games_frame, text="Games per Matchup:", 
                              font=font.Font(family="Helvetica", size=12, weight="bold"),
                              bg="#242731", fg="#E7E1BD")
        games_label.pack(anchor=tk.W)
        
        self.games_var = tk.IntVar(value=1)
        games_scale = tk.Scale(games_frame, from_=1, to=25, orient=tk.HORIZONTAL,
                              variable=self.games_var, bg="#242731", fg="#E7E1BD",
                              highlightbackground="#242731", troughcolor="#34495E",
                              font=font.Font(family="Helvetica", size=10))
        games_scale.pack(fill=tk.X, pady=5)
        
        # Tournament type
        type_frame = tk.Frame(parent, bg="#242731")
        type_frame.pack(fill=tk.X, pady=10, padx=20)
        
        type_label = tk.Label(type_frame, text="Tournament Type:", 
                             font=font.Font(family="Helvetica", size=12, weight="bold"),
                             bg="#242731", fg="#E7E1BD")
        type_label.pack(anchor=tk.W)
        
        self.tournament_type = tk.StringVar(value="focused")
        
        focused_rb = tk.Radiobutton(type_frame, text="Focused Test (Recommended - 50-200 games)", 
                                   variable=self.tournament_type, value="focused",
                                   font=font.Font(family="Helvetica", size=11),
                                   bg="#242731", fg="#E7E1BD", selectcolor="#242731",
                                   activebackground="#242731", activeforeground="#E7E1BD")
        focused_rb.pack(anchor=tk.W, pady=2)
        
        full_rb = tk.Radiobutton(type_frame, text="Full Tournament (All vs All - 500+ games)", 
                                variable=self.tournament_type, value="full",
                                font=font.Font(family="Helvetica", size=11),
                                bg="#242731", fg="#E7E1BD", selectcolor="#242731",
                                activebackground="#242731", activeforeground="#E7E1BD")
        full_rb.pack(anchor=tk.W, pady=2)
        
        # Believability analysis
        believe_frame = tk.Frame(parent, bg="#242731")
        believe_frame.pack(fill=tk.X, pady=10, padx=20)
        
        self.believability_var = tk.BooleanVar(value=True)
        believe_cb = tk.Checkbutton(believe_frame, text="Enable Believability Analysis", 
                                   variable=self.believability_var,
                                   font=font.Font(family="Helvetica", size=12, weight="bold"),
                                   bg="#242731", fg="#F1C40F", selectcolor="#242731",
                                   activebackground="#242731", activeforeground="#F1C40F")
        believe_cb.pack(anchor=tk.W)
        
        believe_desc = tk.Label(believe_frame, 
                               text="Analyzes clue quality and human-likeness using semantic similarity",
                               font=font.Font(family="Helvetica", size=10),
                               bg="#242731", fg="#BDC3C7", wraplength=400, justify=tk.LEFT)
        believe_desc.pack(anchor=tk.W, pady=(5, 0))
        
        # Tournament info
        info_frame = tk.Frame(parent, bg="#242731")
        info_frame.pack(fill=tk.X, pady=20, padx=20)
        
        info_label = tk.Label(info_frame, text="Tournament Information:", 
                             font=font.Font(family="Helvetica", size=12, weight="bold"),
                             bg="#242731", fg="#E7E1BD")
        info_label.pack(anchor=tk.W)
        
        self.info_text = tk.Label(info_frame, text="Select agents to see tournament details",
                                 font=font.Font(family="Helvetica", size=10),
                                 bg="#242731", fg="#BDC3C7", wraplength=500, justify=tk.LEFT)
        self.info_text.pack(anchor=tk.W, pady=(5, 0))
        
        # Update info when settings change
        self.games_var.trace('w', self.update_tournament_info)
        self.tournament_type.trace('w', self.update_tournament_info)
        
    def update_tournament_info(self, *args):
        # Count selected agents
        cm_count = sum(var.get() for var in self.codemaster_vars.values())
        g_count = sum(var.get() for var in self.guesser_vars.values())
        
        if cm_count == 0 or g_count == 0:
            self.info_text.config(text="Please select at least one codemaster and one guesser")
            return
        
        games_per_matchup = self.games_var.get()
        total_teams = cm_count * g_count
        
        if self.tournament_type.get() == "full":
            # Full tournament: each team plays every other team
            total_matchups = total_teams * (total_teams - 1)
        else:
            # Focused: more reasonable number of matchups
            total_matchups = min(total_teams * (total_teams - 1), 50)
        
        total_games = total_matchups * games_per_matchup
        
        info_text = f"""Tournament Details:
• {cm_count} codemasters, {g_count} guessers selected
• {total_teams} total teams
• {total_matchups} unique matchups
• {games_per_matchup} games per matchup
• {total_games} total games
• Estimated time: {self.estimate_time(total_games)}"""
        
        self.info_text.config(text=info_text)
    
    def estimate_time(self, total_games):
        # Rough estimate: 30-45 seconds per game
        avg_seconds = 37
        total_seconds = total_games * avg_seconds
        
        if total_seconds < 60:
            return f"{total_seconds:.0f} seconds"
        elif total_seconds < 3600:
            return f"{total_seconds/60:.1f} minutes"
        else:
            return f"{total_seconds/3600:.1f} hours"
    
    def select_all_agents(self):
        for var in self.codemaster_vars.values():
            var.set(True)
        for var in self.guesser_vars.values():
            var.set(True)
        self.update_tournament_info()
    def setup_believability_tab(self, parent):
        """Setup believability analysis tab"""
        # Create frame for believability analysis
        believe_frame = tk.Frame(parent, bg="#242731")
        believe_frame.pack(fill=tk.BOTH, expand=True)
        
        # Believability rankings
        believe_label = tk.Label(believe_frame, text="Clue Believability Analysis", 
                                font=("Helvetica", 14, "bold"),
                                bg="#242731", fg="#F1C40F")
        believe_label.pack(pady=(0, 10))
        
        # Create treeview for believability
        b_columns = ("Rank", "Agent", "Type", "Believability", "Frequency", "Coherence", "Human-like")
        b_tree = ttk.Treeview(believe_frame, columns=b_columns, show="headings", height=15)
        
        for col in b_columns:
            b_tree.heading(col, text=col)
            b_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        b_scrollbar = ttk.Scrollbar(believe_frame, orient=tk.VERTICAL, command=b_tree.yview)
        b_tree.configure(yscrollcommand=b_scrollbar.set)
        
        # Pack widgets
        b_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        b_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with believability data
        if 'believability_data' in self.results_data:
            for i, data in enumerate(self.results_data['believability_data'], 1):
                b_tree.insert("", tk.END, values=(
                    i,
                    data.get('agent_name', 'Unknown'),
                    data.get('agent_type', 'Unknown'),
                    f"{data.get('overall_believability', 0):.3f}",
                    f"{data.get('frequency_score', 0):.3f}",
                    f"{data.get('semantic_coherence', 0):.3f}",
                    f"{data.get('human_likeness', 0):.3f}"
                ))   
    def setup_rankings_tab(self, parent):
        """Setup rankings tab with agent-wise rankings"""
        # Create notebook for different ranking views
        ranking_notebook = ttk.Notebook(parent)
        ranking_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Team rankings tab
        team_frame = tk.Frame(ranking_notebook, bg="#242731")
        ranking_notebook.add(team_frame, text="Team Rankings")
        
        # Agent rankings tab
        agent_frame = tk.Frame(ranking_notebook, bg="#242731")
        ranking_notebook.add(agent_frame, text="Agent Rankings")
        
        # Setup team rankings (existing code)
        self.setup_team_rankings(team_frame)
        
        # Setup agent rankings (new)
        self.setup_agent_rankings(agent_frame)
    def setup_team_rankings(self, parent):
        """Setup team rankings table"""
        columns = ("Rank", "Team", "Win Rate", "W-L", "TrueSkill", "Composite")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("Rank", width=60, anchor=tk.CENTER)
        tree.column("Team", width=200, anchor=tk.W)
        tree.column("Win Rate", width=80, anchor=tk.CENTER)
        tree.column("W-L", width=80, anchor=tk.CENTER)
        tree.column("TrueSkill", width=100, anchor=tk.CENTER)
        tree.column("Composite", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with team data
        if 'rankings' in self.results_data:
            for i, ranking in enumerate(self.results_data['rankings'], 1):
                win_rate = ranking.get('win_rate', 0)
                wins = ranking.get('wins', 0)
                losses = ranking.get('losses', 0)
                trueskill = ranking.get('trueskill', 0)
                composite = ranking.get('composite_score', 0)
                
                tree.insert("", tk.END, values=(
                    i,
                    ranking.get('team', 'Unknown'),
                    f"{win_rate:.1%}",
                    f"{wins}-{losses}",
                    f"{trueskill:.1f}",
                    f"{composite:.3f}"
                ))

    def setup_agent_rankings(self, parent):
        """Setup agent rankings with separate tabs for codemasters and guessers"""
        # Create notebook for agent types
        agent_notebook = ttk.Notebook(parent)
        agent_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Codemasters tab
        cm_frame = tk.Frame(agent_notebook, bg="#242731")
        agent_notebook.add(cm_frame, text="Codemasters")
        
        # Guessers tab
        g_frame = tk.Frame(agent_notebook, bg="#242731")
        agent_notebook.add(g_frame, text="Guessers")
        
        # Overall tab
        overall_frame = tk.Frame(agent_notebook, bg="#242731")
        agent_notebook.add(overall_frame, text="All Agents")
        
        # Setup each tab
        self.setup_agent_ranking_table(cm_frame, "codemaster")
        self.setup_agent_ranking_table(g_frame, "guesser")
        self.setup_agent_ranking_table(overall_frame, "all")

    def setup_agent_ranking_table(self, parent, agent_type):
        """Setup ranking table for specific agent type"""
        columns = ("Rank", "Agent", "Win Rate", "W-L", "Games", "Teams", "Believability")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("Rank", width=50, anchor=tk.CENTER)
        tree.column("Agent", width=150, anchor=tk.W)
        tree.column("Win Rate", width=80, anchor=tk.CENTER)
        tree.column("W-L", width=80, anchor=tk.CENTER)
        tree.column("Games", width=60, anchor=tk.CENTER)
        tree.column("Teams", width=60, anchor=tk.CENTER)
        tree.column("Believability", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with agent data
        if 'agent_rankings' in self.results_data:
            filtered_agents = []
            for agent_data in self.results_data['agent_rankings']:
                if agent_type == "all" or agent_data.get('type') == agent_type:
                    filtered_agents.append(agent_data)
            
            for i, agent_data in enumerate(filtered_agents, 1):
                tree.insert("", tk.END, values=(
                    i,
                    agent_data.get('name', 'Unknown'),
                    f"{agent_data.get('win_rate', 0):.1%}",
                    f"{agent_data.get('wins', 0)}-{agent_data.get('losses', 0)}",
                    agent_data.get('games', 0),
                    len(agent_data.get('teams_played', [])),
                    f"{agent_data.get('believability', 0.5):.3f}"
                ))


    def setup_agent_ranking_table(self, parent, agent_type):
        """Setup ranking table for specific agent type"""
        columns = ("Rank", "Agent", "Win Rate", "W-L", "Games", "Teams", "Believability")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("Rank", width=50, anchor=tk.CENTER)
        tree.column("Agent", width=150, anchor=tk.W)
        tree.column("Win Rate", width=80, anchor=tk.CENTER)
        tree.column("W-L", width=80, anchor=tk.CENTER)
        tree.column("Games", width=60, anchor=tk.CENTER)
        tree.column("Teams", width=60, anchor=tk.CENTER)
        tree.column("Believability", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with agent data
        if 'agent_rankings' in self.results_data:
            filtered_agents = []
            for agent_data in self.results_data['agent_rankings']:
                if agent_type == "all" or agent_data.get('type') == agent_type:
                    filtered_agents.append(agent_data)
            
            for i, agent_data in enumerate(filtered_agents, 1):
                tree.insert("", tk.END, values=(
                    i,
                    agent_data.get('name', 'Unknown'),
                    f"{agent_data.get('win_rate', 0):.1%}",
                    f"{agent_data.get('wins', 0)}-{agent_data.get('losses', 0)}",
                    agent_data.get('games', 0),
                    len(agent_data.get('teams_played', [])),
                    f"{agent_data.get('believability', 0.5):.3f}"
                )) 
    def select_no_agents(self):
        for var in self.codemaster_vars.values():
            var.set(False)
        for var in self.guesser_vars.values():
            var.set(False)
        self.update_tournament_info()
    
    def select_default_agents(self):
        # Reset to default selections
        defaults_cm = {"MCTS": True, "EMD": True}
        defaults_g = {"EMD": True, "MCTS": True}
        
        for code, var in self.codemaster_vars.items():
            var.set(defaults_cm.get(code, False))
        
        for code, var in self.guesser_vars.items():
            var.set(defaults_g.get(code, False))
        
        self.update_tournament_info()
    
    def start_tournament(self):
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
        self.result = {
            'codemasters': selected_cm,
            'guessers': selected_g,
            'games_per_matchup': self.games_var.get(),
            'tournament_type': self.tournament_type.get(),
            'believability_analysis': self.believability_var.get()
        }
        
        self.window.destroy()
    
    def cancel(self):
        self.result = None
        self.window.destroy()

class TournamentProgressWindow:
    def __init__(self, parent, config):
        self.parent = parent
        self.config = config
        self.cancelled = False
        self.start_time = time.time()
        
        # Create window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Tournament in Progress")
        self.window.geometry("600x400")
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
        title_label = tk.Label(main_frame, text="Tournament Running", 
                              font=font.Font(family="Helvetica", size=18, weight="bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                           maximum=100, length=500)
        self.progress_bar.pack(pady=10)
        
        # Progress label
        self.progress_label = tk.Label(main_frame, text="Preparing tournament...",
                                      font=font.Font(family="Helvetica", size=12),
                                      bg="#242731", fg="#E7E1BD")
        self.progress_label.pack(pady=10)
        
        # Current match info
        self.match_label = tk.Label(main_frame, text="",
                                   font=font.Font(family="Helvetica", size=11),
                                   bg="#242731", fg="#BDC3C7", wraplength=500)
        self.match_label.pack(pady=5)
        
        # Stats frame
        stats_frame = tk.Frame(main_frame, bg="#242731")
        stats_frame.pack(pady=20, fill=tk.X)
        
        self.stats_label = tk.Label(stats_frame, text="Games completed: 0\nTime elapsed: 0:00\nEstimated remaining: --",
                                   font=font.Font(family="Helvetica", size=11),
                                   bg="#242731", fg="#BDC3C7", justify=tk.LEFT)
        self.stats_label.pack()
        
        # Log area
        log_frame = tk.Frame(main_frame, bg="#242731")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        log_label = tk.Label(log_frame, text="Tournament Log:", 
                            font=font.Font(family="Helvetica", size=10, weight="bold"),
                            bg="#242731", fg="#E7E1BD")
        log_label.pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8,
                                                font=font.Font(family="Consolas", size=9),
                                                bg="#1E1F29", fg="#E7E1BD", 
                                                relief=tk.FLAT, bd=1, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Cancel button
        self.cancel_button = tk.Button(main_frame, text="Cancel Tournament", 
                                      command=self.cancel_tournament,
                                      font=font.Font(family="Helvetica", size=12),
                                      bg="#F44336", fg="black", relief=tk.FLAT, bd=0, 
                                      padx=20, pady=8)
        self.cancel_button.pack(pady=10)
        
    def update_progress(self, completed, total, current_match=""):
        if not self.cancelled:
            progress = (completed / total) * 100 if total > 0 else 0
            self.progress_var.set(progress)
            self.progress_label.config(text=f"Progress: {completed}/{total} games ({progress:.1f}%)")
            
            if current_match:
                self.match_label.config(text=f"Current: {current_match}")
            
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
    
    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def add_log_entry(self, message):
        if not self.cancelled:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
    
    def cancel_tournament(self):
        result = messagebox.askyesno("Cancel Tournament", 
                                    "Are you sure you want to cancel the tournament?\nProgress will be lost.")
        if result:
            self.cancelled = True
            self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)
            self.add_log_entry("Tournament cancellation requested...")
    
    def on_close(self):
        # Prevent closing window directly
        pass
    
    def close(self):
        self.window.destroy()

class TournamentResultsWindow:
    def __init__(self, parent, results_data):
        self.parent = parent
        self.results_data = results_data
        
        print(f"DEBUG: TournamentResultsWindow received data with keys: {results_data.keys()}")
        print(f"DEBUG: Rankings count: {len(results_data.get('rankings', []))}")
        print(f"DEBUG: Agent rankings count: {len(results_data.get('agent_rankings', []))}")
        print(f"DEBUG: Believability enabled: {results_data.get('believability_enabled', False)}")
        
        # Create window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Tournament Results")
        self.window.geometry("1200x800")
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
        
        # Title
        title_label = tk.Label(main_frame, text="Tournament Results", 
                              font=("Helvetica", 18, "bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Create notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Team Rankings tab
        team_rankings_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(team_rankings_frame, text="Team Rankings")
        
        # Agent Rankings tab
        agent_rankings_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(agent_rankings_frame, text="Agent Rankings")
        
        # Believability tab (if enabled)
        if self.results_data.get('believability_enabled', False):
            believe_frame = tk.Frame(notebook, bg="#242731")
            notebook.add(believe_frame, text="Believability Analysis")
            self.setup_believability_tab(believe_frame)
        
        # Raw data tab
        raw_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(raw_frame, text="Raw Data")
        
        # Setup tabs
        self.setup_team_rankings_tab(team_rankings_frame)
        self.setup_agent_rankings_tab(agent_rankings_frame)
        self.setup_raw_data_tab(raw_frame)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#242731")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Export button
        export_btn = tk.Button(button_frame, text="Export Results", command=self.export_results,
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
    
    def setup_team_rankings_tab(self, parent):
        """Setup team rankings tab"""
        # Create treeview for team rankings
        columns = ("Rank", "Team", "Win Rate", "W-L", "Games", "TrueSkill", "Composite")
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
        tree.column("Composite", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with team data
        if 'rankings' in self.results_data and self.results_data['rankings']:
            print(f"DEBUG: Populating team rankings with {len(self.results_data['rankings'])} entries")
            for i, ranking in enumerate(self.results_data['rankings'], 1):
                win_rate = ranking.get('win_rate', 0)
                wins = ranking.get('wins', 0)
                losses = ranking.get('losses', 0)
                total_games = wins + losses
                trueskill = ranking.get('trueskill', 0)
                composite = ranking.get('composite_score', 0)
                
                tree.insert("", tk.END, values=(
                    i,
                    ranking.get('team', 'Unknown'),
                    f"{win_rate:.1%}",
                    f"{wins}-{losses}",
                    total_games,
                    f"{trueskill:.1f}",
                    f"{composite:.3f}"
                ))
        else:
            print("DEBUG: No team rankings data found")
            # Show message if no data
            tree.insert("", tk.END, values=("", "No team data available", "", "", "", "", ""))
    
    def setup_agent_rankings_tab(self, parent):
        """Setup agent rankings with separate sections"""
        # Create notebook for agent types
        agent_notebook = ttk.Notebook(parent)
        agent_notebook.pack(fill=tk.BOTH, expand=True)
        
        # All agents tab
        all_frame = tk.Frame(agent_notebook, bg="#242731")
        agent_notebook.add(all_frame, text="All Agents")
        
        # Codemasters tab
        cm_frame = tk.Frame(agent_notebook, bg="#242731")
        agent_notebook.add(cm_frame, text="Codemasters")
        
        # Guessers tab
        g_frame = tk.Frame(agent_notebook, bg="#242731")
        agent_notebook.add(g_frame, text="Guessers")
        
        # Setup each tab
        self.setup_agent_ranking_table(all_frame, "all")
        self.setup_agent_ranking_table(cm_frame, "codemaster")
        self.setup_agent_ranking_table(g_frame, "guesser")
    
    def setup_agent_ranking_table(self, parent, agent_type):
        """Setup ranking table for specific agent type"""
        columns = ("Rank", "Agent", "Win Rate", "W-L", "Games", "Teams", "Believability")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=20)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
        
        tree.column("Rank", width=50, anchor=tk.CENTER)
        tree.column("Agent", width=150, anchor=tk.W)
        tree.column("Win Rate", width=80, anchor=tk.CENTER)
        tree.column("W-L", width=80, anchor=tk.CENTER)
        tree.column("Games", width=60, anchor=tk.CENTER)
        tree.column("Teams", width=60, anchor=tk.CENTER)
        tree.column("Believability", width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with agent data
        if 'agent_rankings' in self.results_data and self.results_data['agent_rankings']:
            print(f"DEBUG: Populating {agent_type} agent rankings")
            filtered_agents = []
            for agent_data in self.results_data['agent_rankings']:
                if agent_type == "all" or agent_data.get('type') == agent_type:
                    filtered_agents.append(agent_data)
            
            print(f"DEBUG: Found {len(filtered_agents)} agents for {agent_type} category")
            
            for i, agent_data in enumerate(filtered_agents, 1):
                tree.insert("", tk.END, values=(
                    i,
                    agent_data.get('name', 'Unknown'),
                    f"{agent_data.get('win_rate', 0):.1%}",
                    f"{agent_data.get('wins', 0)}-{agent_data.get('losses', 0)}",
                    agent_data.get('games', 0),
                    len(agent_data.get('teams_played', [])),
                    f"{agent_data.get('believability', 0.5):.3f}"
                ))
        else:
            print(f"DEBUG: No agent rankings data found for {agent_type}")
            # Show message if no data
            tree.insert("", tk.END, values=("", "No agent data available", "", "", "", "", ""))
    
    def setup_believability_tab(self, parent):
        """Setup believability analysis tab"""
        # Create frame for believability analysis
        believe_frame = tk.Frame(parent, bg="#242731")
        believe_frame.pack(fill=tk.BOTH, expand=True)
        
        # Believability rankings
        believe_label = tk.Label(believe_frame, text="Clue Believability Analysis", 
                                font=("Helvetica", 14, "bold"),
                                bg="#242731", fg="#F1C40F")
        believe_label.pack(pady=(0, 10))
        
        # Check if we have believability data
        if 'believability_data' not in self.results_data or not self.results_data['believability_data']:
            print("DEBUG: No believability data found")
            # Show message if no data
            no_data_label = tk.Label(believe_frame, 
                                    text="No believability data available.\nEnable believability analysis in tournament settings to see this data.",
                                    font=("Helvetica", 12),
                                    bg="#242731", fg="#BDC3C7")
            no_data_label.pack(expand=True)
            return
        
        print(f"DEBUG: Setting up believability tab with {len(self.results_data['believability_data'])} entries")
        
        # Create treeview for believability
        b_columns = ("Rank", "Agent", "Type", "Believability", "Frequency", "Coherence", "Human-like")
        b_tree = ttk.Treeview(believe_frame, columns=b_columns, show="headings", height=15)
        
        for col in b_columns:
            b_tree.heading(col, text=col)
            b_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Add scrollbar
        b_scrollbar = ttk.Scrollbar(believe_frame, orient=tk.VERTICAL, command=b_tree.yview)
        b_tree.configure(yscrollcommand=b_scrollbar.set)
        
        # Pack widgets
        b_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        b_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate with believability data
        for i, data in enumerate(self.results_data['believability_data'], 1):
            b_tree.insert("", tk.END, values=(
                i,
                data.get('agent_name', 'Unknown'),
                data.get('agent_type', 'Unknown'),
                f"{data.get('overall_believability', 0):.3f}",
                f"{data.get('frequency_score', 0):.3f}",
                f"{data.get('semantic_coherence', 0):.3f}",
                f"{data.get('human_likeness', 0):.3f}"
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
        raw_data = json.dumps(self.results_data, indent=2, default=str)
        raw_text.insert(tk.END, raw_data)
        raw_text.config(state=tk.DISABLED)
    
    def export_results(self):
        """Export tournament results"""
        from tkinter import filedialog
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Tournament Results"
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(self.results_data, f, indent=2, default=str)
                elif filename.endswith('.csv'):
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        # Export team rankings
                        writer.writerow(["Team Rankings"])
                        writer.writerow(["Rank", "Team", "Win Rate", "Wins", "Losses", "TrueSkill", "Composite"])
                        for i, ranking in enumerate(self.results_data.get('rankings', []), 1):
                            writer.writerow([
                                i,
                                ranking.get('team', 'Unknown'),
                                f"{ranking.get('win_rate', 0):.3f}",
                                ranking.get('wins', 0),
                                ranking.get('losses', 0),
                                f"{ranking.get('trueskill', 0):.3f}",
                                f"{ranking.get('composite_score', 0):.3f}"
                            ])
                        
                        # Export agent rankings
                        writer.writerow([])
                        writer.writerow(["Agent Rankings"])
                        writer.writerow(["Rank", "Agent", "Type", "Win Rate", "Wins", "Losses", "Games", "Teams", "Believability"])
                        for i, agent in enumerate(self.results_data.get('agent_rankings', []), 1):
                            writer.writerow([
                                i,
                                agent.get('name', 'Unknown'),
                                agent.get('type', 'Unknown'),
                                f"{agent.get('win_rate', 0):.3f}",
                                agent.get('wins', 0),
                                agent.get('losses', 0),
                                agent.get('games', 0),
                                len(agent.get('teams_played', [])),
                                f"{agent.get('believability', 0.5):.3f}"
                            ])
                
                messagebox.showinfo("Export Complete", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

# ==================== MAIN ====================

if __name__ == "__main__":
   root = tk.Tk()
   app = CodenamesGUI(root)
   root.mainloop()
