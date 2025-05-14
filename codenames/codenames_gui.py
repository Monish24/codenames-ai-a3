import tkinter as tk
from tkinter import scrolledtext, font, messagebox
import sys
import threading
import time
import subprocess
import queue
import re
import os

class CodenamesGUI:
    def __init__(self, root):
        self.guessed_words = set()  # Track words that have been guessed
        self.word_results = []  # Track results with team ownership
        self.board_initialized = False  # Flag to track if board is fully initialized
        self.seen_clues = set()  
        self.seen_guesses = set()
        self.seen_results = set()
        self.root = root
        self.root.title("Codenames AI Game")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2C3E50")
        
        # Initialize output queue
        self.output_queue = queue.Queue()
        
        self.setup_fonts()
        self.create_widgets()
        self.setup_stdout_redirection()
        
        self.game_process = None
        self.game_running = False
        
        # Game state variables
        self.board_words = []
        self.key_grid = []
        self.current_turn = "Red"
        self.current_clue = ""
        self.current_clue_num = 0
        self.current_guess = ""
        
        # Game statistics
        self.red_words_found = 0
        self.blue_words_found = 0
        self.civilian_words_found = 0
        self.assassin_found = False
        self.spymaster_mode = True
        # Start checking the queue
        self.root.after(100, self.check_output_queue)
    
    def setup_fonts(self):
        """Set up fonts for the GUI"""
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.board_font = font.Font(family="Helvetica", size=12, weight="bold")
        self.log_font = font.Font(family="Helvetica", size=11)
        self.button_font = font.Font(family="Helvetica", size=12)
    
    def create_widgets(self):
        """Create and organize all GUI elements with improved styling"""
        # Main frame
        main_frame = tk.Frame(self.root, bg="#242731")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="Codenames AI Game", 
                            font=font.Font(family="Helvetica", size=22, weight="bold"), 
                            bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=10)
        
        # Game controls frame
        controls_frame = tk.Frame(main_frame, bg="#242731")
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Start game button
        self.start_button = tk.Button(controls_frame, text="Start Game", 
                                    command=self.start_game, 
                                    font=self.button_font,
                                    bg="#4CAF50", fg="black",  # Changed to black text
                                    relief=tk.FLAT, bd=0, padx=10, width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop game button
        self.stop_button = tk.Button(controls_frame, text="Stop Game", 
                                    command=self.stop_game, 
                                    font=self.button_font,
                                    bg="#F44336", fg="black",  # Changed to black text
                                    relief=tk.FLAT, bd=0, padx=10, width=15, 
                                    state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Debug button
        self.debug_button = tk.Button(controls_frame, text="Show Debug Info", 
                                    command=self.show_debug_info, 
                                    font=self.button_font,
                                    bg="#FF9800", fg="black",  # Changed to black text
                                    relief=tk.FLAT, bd=0, padx=10, width=15)
        self.debug_button.pack(side=tk.LEFT, padx=5)
        
        # Spymaster toggle button
        self.spymaster_button = tk.Button(controls_frame, text="Spymaster View", 
                                        command=self.toggle_spymaster_view, 
                                        font=self.button_font,
                                        bg="#673AB7", fg="black",  # Changed to black text
                                        relief=tk.FLAT, bd=0, padx=10, width=18)
        self.spymaster_button.pack(side=tk.LEFT, padx=5)
        self.spymaster_mode = False
        
        # Game configuration frame
        config_frame = tk.Frame(main_frame, bg="#242731")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Team configuration section - 4 dropdowns
        role_options = ["Simple Embedded", "MCTS", "RL", "Hybrid1", "Hybrid2", "Human"]

        # Red team frame
        red_team_frame = tk.Frame(config_frame, bg="#242731")
        red_team_frame.pack(side=tk.LEFT, padx=10)

        # Red team label
        red_team_label = tk.Label(red_team_frame, text="Red Team:", 
                                font=self.board_font, bg="#242731", fg="#C13A37")
        red_team_label.pack(anchor=tk.W)

        # Red Codemaster dropdown
        red_cm_frame = tk.Frame(red_team_frame, bg="#242731")
        red_cm_frame.pack(fill=tk.X, pady=2)

        red_cm_label = tk.Label(red_cm_frame, text="Codemaster:", 
                            font=self.board_font, bg="#242731", fg="#E7E1BD")
        red_cm_label.pack(side=tk.LEFT, padx=2)

        self.red_codemaster = tk.StringVar(value="MCTS")
        red_cm_dropdown = tk.OptionMenu(red_cm_frame, self.red_codemaster, *role_options)
        red_cm_dropdown.config(font=self.board_font, bg="#34495E", fg="white", width=12)
        red_cm_dropdown.pack(side=tk.LEFT, padx=2)

        # Red Guesser dropdown
        red_g_frame = tk.Frame(red_team_frame, bg="#242731")
        red_g_frame.pack(fill=tk.X, pady=2)

        red_g_label = tk.Label(red_g_frame, text="Guesser:", 
                            font=self.board_font, bg="#242731", fg="#E7E1BD")
        red_g_label.pack(side=tk.LEFT, padx=2)

        self.red_guesser = tk.StringVar(value="Simple Embedded")
        red_g_dropdown = tk.OptionMenu(red_g_frame, self.red_guesser, *role_options)
        red_g_dropdown.config(font=self.board_font, bg="#34495E", fg="white", width=12)
        red_g_dropdown.pack(side=tk.LEFT, padx=2)

        # Blue team frame
        blue_team_frame = tk.Frame(config_frame, bg="#242731")
        blue_team_frame.pack(side=tk.LEFT, padx=10)

        # Blue team label
        blue_team_label = tk.Label(blue_team_frame, text="Blue Team:", 
                                font=self.board_font, bg="#242731", fg="#4989C5")
        blue_team_label.pack(anchor=tk.W)

        # Blue Codemaster dropdown
        blue_cm_frame = tk.Frame(blue_team_frame, bg="#242731")
        blue_cm_frame.pack(fill=tk.X, pady=2)

        blue_cm_label = tk.Label(blue_cm_frame, text="Codemaster:", 
                                font=self.board_font, bg="#242731", fg="#E7E1BD")
        blue_cm_label.pack(side=tk.LEFT, padx=2)

        self.blue_codemaster = tk.StringVar(value="MCTS")
        blue_cm_dropdown = tk.OptionMenu(blue_cm_frame, self.blue_codemaster, *role_options)
        blue_cm_dropdown.config(font=self.board_font, bg="#34495E", fg="white", width=12)
        blue_cm_dropdown.pack(side=tk.LEFT, padx=2)

        # Blue Guesser dropdown
        blue_g_frame = tk.Frame(blue_team_frame, bg="#242731")
        blue_g_frame.pack(fill=tk.X, pady=2)

        blue_g_label = tk.Label(blue_g_frame, text="Guesser:", 
                            font=self.board_font, bg="#242731", fg="#E7E1BD")
        blue_g_label.pack(side=tk.LEFT, padx=2)

        self.blue_guesser = tk.StringVar(value="Simple Embedded")
        blue_g_dropdown = tk.OptionMenu(blue_g_frame, self.blue_guesser, *role_options)
        blue_g_dropdown.config(font=self.board_font, bg="#34495E", fg="white", width=12)
        blue_g_dropdown.pack(side=tk.LEFT, padx=2)
        
        # Seed input
        seed_label = tk.Label(config_frame, text="Seed:", 
                            font=self.board_font, bg="#242731", fg="#E7E1BD")
        seed_label.pack(side=tk.LEFT, padx=5)
        
        self.seed_entry = tk.Entry(config_frame, font=self.board_font, width=10, 
                                bg="#34495E", fg="white", insertbackground="white")
        self.seed_entry.insert(0, "42")
        self.seed_entry.pack(side=tk.LEFT, padx=5)
        
        # Content pane - split into board and game info (using a PanedWindow)
        content_paned = tk.PanedWindow(main_frame, bg="#242731", orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Game board frame (left side)
        board_frame = tk.Frame(content_paned, bg="#1E1F29", bd=0)
        
        # Apply panel styling with border and shadow
        board_frame.configure(highlightbackground="#1A1A1A", highlightthickness=1)
        shadow_frame = tk.Frame(content_paned, bg="#0A0A0A")
        shadow_frame.place(in_=board_frame, relx=1.002, rely=1.002, relwidth=1, relheight=1)
        
        # Board title and current team
        board_title_frame = tk.Frame(board_frame, bg="#1E1F29")
        board_title_frame.pack(fill=tk.X, pady=5)
        
        board_title = tk.Label(board_title_frame, text="Game Board", 
                            font=self.title_font, bg="#1E1F29", fg="#E7E1BD")
        board_title.pack(side=tk.LEFT, padx=10)
        
        self.turn_label = tk.Label(board_title_frame, text="", 
                                font=self.board_font, bg="#1E1F29", fg="#F1C40F")
        self.turn_label.pack(side=tk.RIGHT, padx=10)
        
        # Board grid
        self.board_frame = tk.Frame(board_frame, bg="#1E1F29", padx=15, pady=15)
        self.board_frame.pack(fill=tk.BOTH, expand=True)
        
        self.word_tiles = []
        for i in range(5):
            row = []
            for j in range(5):
                tile_frame = tk.Frame(self.board_frame, bg="#1E1F29", padx=3, pady=3)
                tile_frame.grid(row=i, column=j, padx=5, pady=5, sticky="nsew")
                
                tile = tk.Label(tile_frame, text="", width=10, height=3,
                            font=font.Font(family="Helvetica", size=12, weight="bold"),
                            bg="#E7E1BD", fg="black",
                            relief=tk.RAISED, bd=2)
                tile.pack(fill=tk.BOTH, expand=True)
                row.append(tile)
            self.word_tiles.append(row)
        
        # Configure grid weights
        for i in range(5):
            self.board_frame.grid_rowconfigure(i, weight=1)
            self.board_frame.grid_columnconfigure(i, weight=1)
        
        info_frame = tk.Frame(content_paned, bg="#1E1F29", bd=0)
        
        info_frame.configure(highlightbackground="#1A1A1A", highlightthickness=1)
        shadow_frame2 = tk.Frame(content_paned, bg="#0A0A0A")
        shadow_frame2.place(in_=info_frame, relx=1.002, rely=1.002, relwidth=1, relheight=1)
        
        content_paned.add(board_frame, stretch="always", width=600)  # Board gets more space
        content_paned.add(info_frame, stretch="always", width=400)   # Info panel gets less space
        
        board_frame.lift()
        info_frame.lift()
        
        actions_frame = tk.Frame(info_frame, bg="#1E1F29")
        actions_frame.pack(fill=tk.X, pady=5, padx=10)
        
        self.clue_frame = tk.Frame(actions_frame, bg="#1E1F29", bd=1, relief=tk.GROOVE)
        self.clue_frame.pack(fill=tk.X, pady=5)
        
        clue_title = tk.Label(self.clue_frame, text="Current Clue:", 
                            font=self.board_font, bg="#1E1F29", fg="#E7E1BD")
        clue_title.pack(anchor=tk.W, padx=10, pady=5)
        
        self.clue_label = tk.Label(self.clue_frame, text="Waiting for clue...", 
                                font=self.title_font, bg="#1E1F29", fg="#F1C40F")
        self.clue_label.pack(padx=10, pady=5)
        
        self.guess_frame = tk.Frame(actions_frame, bg="#1E1F29", bd=1, relief=tk.GROOVE)
        self.guess_frame.pack(fill=tk.X, pady=5)
        
        guess_title = tk.Label(self.guess_frame, text="Latest Guess:", 
                            font=self.board_font, bg="#1E1F29", fg="#E7E1BD")
        guess_title.pack(anchor=tk.W, padx=10, pady=5)
        
        self.guess_label = tk.Label(self.guess_frame, text="Waiting for guess...", 
                                font=self.title_font, bg="#1E1F29", fg="white")
        self.guess_label.pack(padx=10, pady=5)
        
        # Score display
        self.score_frame = tk.Frame(actions_frame, bg="#1E1F29", bd=0, relief=tk.FLAT)
        self.score_frame.pack(fill=tk.X, pady=10)
        
        score_title = tk.Label(self.score_frame, text="Score:", 
                            font=self.board_font, bg="#1E1F29", fg="#E7E1BD")
        score_title.pack(anchor=tk.W, padx=10, pady=5)
        
        score_display = tk.Frame(self.score_frame, bg="#1E1F29")
        score_display.pack(fill=tk.X, padx=10, pady=5)
        
        # Red team score
        red_frame = tk.Frame(score_display, bg="#1E1F29")
        red_frame.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        red_label = tk.Label(red_frame, text="Red", 
                            font=font.Font(family="Helvetica", size=14, weight="bold"), 
                            bg="#1E1F29", fg="#C13A37")
        red_label.pack()
        
        self.red_score = tk.Label(red_frame, text="0/9", 
                                font=font.Font(family="Helvetica", size=20, weight="bold"), 
                                bg="#1E1F29", fg="#C13A37")
        self.red_score.pack()
        
        # Blue team score
        blue_frame = tk.Frame(score_display, bg="#1E1F29")
        blue_frame.pack(side=tk.RIGHT, expand=True, fill=tk.X)
        
        blue_label = tk.Label(blue_frame, text="Blue", 
                            font=font.Font(family="Helvetica", size=14, weight="bold"), 
                            bg="#1E1F29", fg="#4989C5")
        blue_label.pack()
        
        self.blue_score = tk.Label(blue_frame, text="0/8", 
                                font=font.Font(family="Helvetica", size=20, weight="bold"),
                                bg="#1E1F29", fg="#4989C5")
        self.blue_score.pack()
        
        # Game log panel
        log_frame = tk.Frame(info_frame, bg="#1E1F29", bd=0)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)
        
        log_title = tk.Label(log_frame, text="Game History", 
                            font=self.title_font, bg="#1E1F29", fg="#E7E1BD")
        log_title.pack(pady=5)
        
        # Game log with improved styling
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                font=self.log_font,
                                                bg="#242731", fg="#E7E1BD", 
                                                relief=tk.FLAT, bd=1, 
                                                padx=10, pady=10,
                                                wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text tags with enhanced colors
        self.log_text.tag_configure("red_turn", foreground="#C13A37", 
                                    font=font.Font(family="Helvetica", size=12, weight="bold"))
        self.log_text.tag_configure("blue_turn", foreground="#4989C5", 
                                    font=font.Font(family="Helvetica", size=12, weight="bold"))
        self.log_text.tag_configure("red_clue", foreground="#C13A37")
        self.log_text.tag_configure("blue_clue", foreground="#4989C5")
        self.log_text.tag_configure("red_guess", foreground="#C13A37")
        self.log_text.tag_configure("blue_guess", foreground="#4989C5")
        self.log_text.tag_configure("separator", foreground="#666666")
        self.log_text.tag_configure("win", foreground="#FFD700", 
                                    font=font.Font(family="Helvetica", size=16, weight="bold"))
        self.log_text.tag_configure("game_event", foreground="#E7E1BD", 
                                    font=font.Font(family="Helvetica", size=12, weight="bold"))
        self.log_text.tag_configure("debug", foreground="#E67E22")
        self.setup_button_hover()
    
    def setup_button_hover(self):
        """Setup hover effects for buttons"""
        def on_enter(e, button, hover_color):
            button['background'] = hover_color
            
        def on_leave(e, button, original_color):
            button['background'] = original_color
        
        # Set up hover effects for each button
        buttons = [
            (self.start_button, "#4CAF50", "#81C784"),  # Green, lighter green
            (self.stop_button, "#F44336", "#E57373"),   # Red, lighter red
            (self.debug_button, "#FF9800", "#FFB74D"),  # Orange, lighter orange
            (self.spymaster_button, "#673AB7", "#9575CD")  # Purple, lighter purple
        ]
        
        for button, original, hover in buttons:
            button.bind("<Enter>", lambda e, b=button, h=hover: on_enter(e, b, h))
            button.bind("<Leave>", lambda e, b=button, o=original: on_leave(e, b, o))

    def setup_stdout_redirection(self):
        """Redirect stdout to capture game output"""
        self.original_stdout = sys.stdout
        
        class StdoutRedirector:
            
            def __init__(self, queue):
                self.queue = queue
                self.original_stdout = None
            
            def write(self, string):
                self.queue.put(string)
                if self.original_stdout:
                    self.original_stdout.write(string)
            
            def flush(self):
                if self.original_stdout:
                    self.original_stdout.flush()
        
        self.stdout_redirector = StdoutRedirector(self.output_queue)
        self.stdout_redirector.original_stdout = self.original_stdout
        sys.stdout = self.stdout_redirector
    
    def check_output_queue(self):
        """Check for new output in the queue"""
        try:
            while not self.output_queue.empty():
                output = self.output_queue.get_nowait()
                self.process_output(output)
        except Exception as e:
            self.add_log_entry(f"Error processing output: {str(e)}", "debug")
        
        # Schedule the next check
        self.root.after(100, self.check_output_queue)
    
    def process_output(self, output):
        """Process captured output and update the GUI accordingly"""
        # Only log for debugging purposes if needed
        debug_mode = False
        if debug_mode and any(marker in output for marker in [
            "BOARD", "RED TEAM TURN", "BLUE TEAM TURN", 
            "CLUE:", "clue is", "guess_answer", "Guessing:", "RESULT:"
        ]):
            self.add_log_entry(f"OUTPUT: {output.strip()}", "debug")
        
        # Parse the output for game state updates
        self.parse_output(output)

    def parse_output(self, output):
        """Parse output to update game state"""
        # Check for board updates
        if "BOARD" in output:
            board_updated = self.try_extract_board(output)
            if board_updated and not self.board_initialized:
                self.board_initialized = True
                # If first board update, set spymaster mode on
                self.spymaster_mode = True
                self.spymaster_button.config(text="Player View")
                # Update display to show all colors
                self.update_board_display()
                self.add_log_entry("Board initialized!", "game_event")
        
        # Check for team turn - avoid duplicates
        if "RED TEAM TURN" in output and self.current_turn != "Red":
            self.current_turn = "Red"
            self.turn_label.config(text="Red Team's Turn", fg="#C13A37")
            self.add_log_entry("Red Team's Turn", "red_turn")
            self.add_log_entry("-------------------------", "separator")
        
        elif "BLUE TEAM TURN" in output and self.current_turn != "Blue":
            self.current_turn = "Blue"
            self.turn_label.config(text="Blue Team's Turn", fg="#4989C5")
            self.add_log_entry("Blue Team's Turn", "blue_turn")
            self.add_log_entry("-------------------------", "separator")
        
        # Check for clue - improved pattern matching & avoid duplicates
        clue_match = re.search(r"(?:clue is:|The clue is:|CLUE:|CODEMASTER CLUE:)\s*(\w+)\s+(\d+)", output, re.IGNORECASE)
        if clue_match:
            clue_word = clue_match.group(1)
            clue_num = clue_match.group(2)
            
            # Create a unique key for this clue
            clue_key = f"{clue_word}_{clue_num}_{self.current_turn}"
            
            # Only process if we haven't seen this clue before
            if clue_key not in self.seen_clues:
                self.seen_clues.add(clue_key)
                self.update_clue(clue_word, clue_num)
        
        # Check for guess answer - improved pattern & avoid duplicates
        guess_pattern = re.search(r"(?:guess_answer\s*=\s*|Guessing:\s*)([A-Z]+)", output, re.IGNORECASE)
        if guess_pattern:
            guess_word = guess_pattern.group(1)
            
            # Create a unique key for this guess
            guess_key = f"{guess_word}_{self.current_turn}"
            
            # Only process if we haven't seen this guess before
            if guess_key not in self.seen_guesses:
                self.seen_guesses.add(guess_key)
                self.update_guess(guess_word)
        
        # Check for results - avoid duplicates
        result_pattern = re.search(r"RESULT:\s*(.*)", output)
        if result_pattern:
            result_text = result_pattern.group(1).strip()
            
            # Create a unique key for this result
            result_key = f"{result_text}_{self.current_turn}"
            
            # Only process if we haven't seen this result before
            if result_key not in self.seen_results:
                self.seen_results.add(result_key)
                team_color = "red_guess" if self.current_turn == "Red" else "blue_guess"
                self.add_log_entry(f"Result: {result_text}", team_color)
                
                # Extract the word from the result
                word_match = re.search(r"([A-Z]+)\s+is", result_text, re.IGNORECASE)
                if word_match:
                    result_word = word_match.group(1)
                    
                    # Determine the team
                    if "Red team word" in result_text:
                        self.word_results.append((result_word, "red"))
                    elif "Blue team word" in result_text:
                        self.word_results.append((result_word, "blue"))
                    elif "Civilian" in result_text:
                        self.word_results.append((result_word, "civilian"))
                    elif "Assassin" in result_text:
                        self.word_results.append((result_word, "assassin"))
                        
                    # Update the tile for this word
                    self.update_tile_for_word(result_word)
        
        # Check for win condition
        if "Red Team Wins" in output:
            self.add_log_entry("🏆 Red Team Wins! 🏆", "win")
            self.turn_label.config(text="Game Over - Red Wins!", fg="#C13A37")
        elif "Blue Team Wins" in output:
            self.add_log_entry("🏆 Blue Team Wins! 🏆", "win")
            self.turn_label.config(text="Game Over - Blue Wins!", fg="#4989C5")

    def add_log_entry(self, text, tag=None):
        """Add an entry to the game log"""
        self.log_text.config(state=tk.NORMAL)
        
        # Add the text with appropriate tag if provided
        if tag:
            self.log_text.insert(tk.END, text + "\n", tag)
        else:
            self.log_text.insert(tk.END, text + "\n")
            
        # Scroll to the end
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def get_guess_result_emoji(self, guess_word):
        """Determine the appropriate emoji based on the guess result"""
        # Look for the word on the board
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                if tile.cget("text").replace("*", "") == guess_word:
                    bg_color = tile.cget("bg")
                    if bg_color == "#E74C3C":  # Red
                        return "✅" if self.current_turn == "Red" else "❌"
                    elif bg_color == "#3498DB":  # Blue
                        return "✅" if self.current_turn == "Blue" else "❌"
                    elif bg_color == "#7F8C8D":  # Civilian
                        return "⚪"
                    elif bg_color == "#2C3E50":  # Assassin
                        return "☠️"
        return "Agent"  # Unknown result
    

    def try_extract_board(self, output):
        """Attempt to extract both the key grid and board words to properly map teams to words"""
        # Store extraction results
        key_grid = []
        board_words = []
        
        # Track if this is our first successful extraction
        first_extraction = not self.board_initialized
        
        # Look for the complete pattern first (both KEY and BOARD sections together)
        full_pattern = re.search(r"KEY[_=]+\s*(.*?)[_=]+\s*BOARD[_=]+\s*(.*?)[_=]+", output, re.DOTALL)
        if full_pattern:
            # Extract key section
            key_section = full_pattern.group(1).strip()
            key_lines = [line.strip() for line in key_section.split('\n') if line.strip()]
            
            # Extract board section
            board_section = full_pattern.group(2).strip()
            board_lines = [line.strip() for line in board_section.split('\n') if line.strip()]
            
            # Process key grid - extract team names
            for line in key_lines:
                if not any(team in line.lower() for team in ["red", "blue", "civilian", "assassin"]):
                    continue
                    
                teams_in_line = []
                for team in re.findall(r'\b(Red|Blue|Civilian|Assassin)\b', line, re.IGNORECASE):
                    teams_in_line.append(team.lower())
                
                if teams_in_line:
                    key_grid.append(teams_in_line)
            
            for line in board_lines:
                if not re.search(r'[A-Za-z]{3,}', line):
                    continue
                    
                words_in_line = []
                for word in re.findall(r'\b([A-Za-z]{3,})\b', line):
                    if word.lower() not in ('red', 'blue', 'civilian', 'assassin'):
                        words_in_line.append(word.upper())
                
                if words_in_line:
                    board_words.append(words_in_line)
        
        if key_grid and board_words:
            flat_key = [team for row in key_grid for team in row]
            flat_words = [word for row in board_words for word in row]
            
            if first_extraction:
                self.word_results = []
            
            # Create a map of existing word->team mappings for quick lookup
            existing_mappings = {word.upper(): team for word, team in self.word_results}
            
            # Map each word to its team
            for i in range(min(len(flat_key), len(flat_words))):
                word = flat_words[i]
                team = flat_key[i]
                
                # Only add if this is a new word or first extraction
                if word not in existing_mappings:
                    self.word_results.append((word, team))
                    print(f"Mapped {word} to {team}")
                    
                    # Mark assassin specially
                    if team == "assassin":
                        print(f"Found assassin card: {word}")
            
            # Update the board display with all words
            # Include both newly extracted words and previously mapped words
            all_words = list(set(flat_words) | set(existing_mappings.keys()))
            
            if all_words:
                # Ensure we have the complete set for the board
                self.update_board(all_words)
                
                # Make sure spymaster view is active to show colors
                if first_extraction:
                    # Force to spymaster view on first extraction
                    self.spymaster_mode = True
                    self.spymaster_button.config(text="Player View")
                
                # Always update the display with current spymaster setting
                self.update_board_display()
                
                # Mark as initialized after first successful extraction
                if first_extraction:
                    self.board_initialized = True
                    self.add_log_entry("Board initialized in Spymaster view!", "game_event")
                
                return True
        
        return False

    def update_board(self, words):
        """Update the board display with the current words"""
        if not words:
            return
            
        # Clean up words list
        clean_words = []
        for w in words:
            clean_word = re.sub(r'[^a-zA-Z0-9]', '', w)
            if clean_word:
                clean_words.append(clean_word.upper())  # Convert to uppercase for consistency
        
        # Ensure exactly 25 words for the 5x5 grid
        if len(clean_words) < 25:
            print(f"Warning: Only found {len(clean_words)} words, expected 25")
            # Pad with placeholder text if needed
            clean_words.extend(["WORD"] * (25 - len(clean_words)))
        elif len(clean_words) > 25:
            print(f"Warning: Found {len(clean_words)} words, expected 25, truncating")
            clean_words = clean_words[:25]
        
        # Debug output to check all words
        print("All 25 words:", clean_words)
        
        # Update board tiles
        for i in range(5):
            for j in range(5):
                idx = i*5 + j
                word = clean_words[idx]
                tile = self.word_tiles[i][j]
                
                # Always set the word text
                tile.config(text=word)
                
                # Check if this word has been guessed
                is_guessed = word in self.guessed_words
                
                # Reset colors to neutral initially
                tile.config(bg="#E7E1BD", fg="black", relief=tk.RAISED if not is_guessed else tk.SUNKEN)
        
        # After updating all tiles, apply the current view mode
        self.update_board_display()

    def update_clue(self, clue_word, clue_num):
        """Update the current clue display and log"""
        self.current_clue = clue_word
        self.current_clue_num = clue_num
        
        # Update the clue label
        clue_text = f"{clue_word.upper()} ({clue_num})"
        
        if self.current_turn == "Red":
            self.clue_label.config(text=clue_text, fg="#E74C3C")
            self.add_log_entry(f"Clue: {clue_text}", "red_clue")
        else:
            self.clue_label.config(text=clue_text, fg="#3498DB")
            self.add_log_entry(f"Clue: {clue_text}", "blue_clue")
    
    def update_guess(self, guess_word):
        """Update the current guess display and log"""
        self.current_guess = guess_word
        self.guess_label.config(text=guess_word.upper())
        
        # Get result emoji (will be updated when result is processed)
        emoji = "❓"  # Start with unknown
        
        # Check if we already know the result
        for result_word, result_team in self.word_results:
            if result_word.upper() == guess_word.upper():
                # We know the team already
                if result_team == "red":
                    emoji = "✅" if self.current_turn == "Red" else "❌"
                elif result_team == "blue":
                    emoji = "✅" if self.current_turn == "Blue" else "❌"
                elif result_team == "civilian":
                    emoji = "⚪"
                elif result_team == "assassin":
                    emoji = "☠️"
                break
        
        # Add to log with appropriate team color
        if self.current_turn == "Red":
            self.add_log_entry(f"{emoji} Guessed: {guess_word.upper()}", "red_guess")
        else:
            self.add_log_entry(f"{emoji} Guessed: {guess_word.upper()}", "blue_guess")

    def update_tile_for_word(self, word):
        """Update the tile for a specific word"""
        word = word.upper()
        
        # Try to find the team for this word based on results
        team = "unknown"
        for result_word, result_team in self.word_results:
            if result_word.upper() == word:
                team = result_team
                break
        
        # Flag to track if this is the first time revealing this word
        first_reveal = word not in self.guessed_words
        
        # Add to guessed words
        self.guessed_words.add(word)
        
        # Update the tile
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                if tile.cget("text").upper() == word:
                    # Set visual appearance based on team
                    if team == "red":
                        tile.config(bg="#C13A37", fg="white", relief=tk.GROOVE, bd=2)
                        # Only update score once
                        if first_reveal:
                            self.red_words_found += 1
                            self.red_score.config(text=f"{self.red_words_found}/9")
                    elif team == "blue":
                        tile.config(bg="#4989C5", fg="white", relief=tk.GROOVE, bd=2)
                        # Only update score once
                        if first_reveal:
                            self.blue_words_found += 1
                            self.blue_score.config(text=f"{self.blue_words_found}/8")
                    elif team == "civilian":
                        tile.config(bg="#DCD6B0", fg="black", relief=tk.GROOVE, bd=2)
                        # Only update score once
                        if first_reveal:
                            self.civilian_words_found += 1
                    elif team == "assassin":
                        tile.config(bg="#2C2C2E", fg="white", relief=tk.GROOVE, bd=2)
                        # Only update score once
                        if first_reveal:
                            self.assassin_found = True
                    break

    def toggle_spymaster_view(self):
        """Toggle between player view and spymaster view"""
        # Toggle the mode
        self.spymaster_mode = not self.spymaster_mode
                
        # Update the button text
        if self.spymaster_mode:
            self.spymaster_button.config(text="Player View", bg="#673AB7", fg="white")
        else:
            self.spymaster_button.config(text="Spymaster View", bg="#673AB7", fg="white")
        
        # Update the board display based on the new mode
        self.update_board_display()

    def update_board_display(self):
        """Update the board display based on current spymaster mode"""
        # Improved colors based on codenames.game
        red_color = "#C13A37"      # Darker red
        blue_color = "#4989C5"     # Softer blue
        civilian_color = "#DCD6B0" # Tan/beige
        assassin_color = "#2C2C2E" # Almost black (dark gray)
        neutral_color = "#E7E1BD"  # Light tan card color
        
        # Update all tiles based on current knowledge and mode
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                word = tile.cget("text")
                
                if not word or word == "Loading...":
                    continue
                    
                # Check if this word has been guessed
                is_guessed = word.upper() in self.guessed_words
                
                # Determine the team for this word
                team = "unknown"
                for result_word, result_team in self.word_results:
                    if result_word.upper() == word.upper():
                        team = result_team
                        break
                
                # Set the appropriate style based on team and current view mode
                if self.spymaster_mode:
                    # Spymaster view - show all team colors
                    if team == "red":
                        tile.config(bg=red_color, fg="white")
                    elif team == "blue":
                        tile.config(bg=blue_color, fg="white")
                    elif team == "civilian":
                        tile.config(bg=civilian_color, fg="black")
                    elif team == "assassin":
                        # Make assassin card very obvious
                        tile.config(bg=assassin_color, fg="white")
                    else:
                        # Unknown team - use neutral color
                        tile.config(bg=neutral_color, fg="black")
                else:
                    # Player view - only show guessed cards with color
                    if is_guessed:
                        # Revealed cards show their team color
                        if team == "red":
                            tile.config(bg=red_color, fg="white")
                        elif team == "blue":
                            tile.config(bg=blue_color, fg="white")
                        elif team == "civilian":
                            tile.config(bg=civilian_color, fg="black")
                        elif team == "assassin":
                            tile.config(bg=assassin_color, fg="white")
                    else:
                        # Unrevealed cards look like neutral cards
                        tile.config(bg=neutral_color, fg="black")
                
                # Always update the relief based on guessed status
                if is_guessed:
                    tile.config(relief=tk.SUNKEN, bd=3)
                else:
                    tile.config(relief=tk.RAISED, bd=2)

    def show_debug_info(self):
        """Show debug information about the game state"""
        debug_info = [
            "=== DEBUG INFORMATION ===",
            f"Current directory: {os.getcwd()}",
            f"Game running: {self.game_running}",
            f"Current turn: {self.current_turn}",
            f"Clue: {self.current_clue} ({self.current_clue_num})",
            f"Guess: {self.current_guess}",
            f"Red words found: {self.red_words_found}/9",
            f"Blue words found: {self.blue_words_found}/8",
            f"Civilian words found: {self.civilian_words_found}",
            f"Assassin found: {self.assassin_found}",
            "Board state:",
        ]
        
        # Add board state
        for row in self.word_tiles:
            board_row = []
            for tile in row:
                board_row.append(tile.cget("text"))
            debug_info.append(" | ".join(board_row))
        
        # Log the debug info
        for line in debug_info:
            self.add_log_entry(line, "debug")
        
        # Check for required files
        req_files = [
            "game.py", 
            "run_game.py", 
            os.path.join("players", "codemaster.py"),
            os.path.join("players", "guesser.py"),
            os.path.join("players", "codemasterMCTS.py"),
            os.path.join("players", "guesserEMD.py"),
            os.path.join("players", "cm_wordlist.txt"),
        ]
        
        self.add_log_entry("Checking for required files:", "debug")
        for file in req_files:
            exists = os.path.exists(file)
            self.add_log_entry(f"  {file}: {'✅' if exists else '❌'}", "debug")
    
    def start_game(self):
        """Start the game with the selected configuration"""
        if self.game_running:
            return
        
        self.game_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Reset game state
        self.board_words = []
        self.key_grid = []
        self.current_turn = "Red"
        self.current_clue = ""
        self.current_clue_num = 0
        self.current_guess = ""
        
        # Reset tracking variables
        self.guessed_words = set()
        self.word_results = []
        self.board_initialized = False
        self.seen_clues = set()
        self.seen_guesses = set()
        self.seen_results = set()
        
        # Reset statistics
        self.red_words_found = 0
        self.blue_words_found = 0
        self.civilian_words_found = 0
        self.assassin_found = False
        
        # Reset display
        self.turn_label.config(text="")
        self.clue_label.config(text="Waiting for clue...")
        self.guess_label.config(text="Waiting for guess...")
        self.red_score.config(text="0/9")
        self.blue_score.config(text="0/8")
        
        # Clear the log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Reset all tiles to loading state
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                tile.config(text="Loading...", bg="#E7E1BD", fg="#2C3E50", relief=tk.RAISED, bd=2)
        
        # Get the selected team configurations
        red_cm = self.red_codemaster.get()
        red_g = self.red_guesser.get()
        blue_cm = self.blue_codemaster.get()
        blue_g = self.blue_guesser.get()
        
        # Get the seed
        seed = self.seed_entry.get() or "42"
        
        # Check if required files exist
        players_dir = "players"
        if not os.path.exists(players_dir):
            self.add_log_entry(f"Error: 'players' directory not found", "debug")
            messagebox.showerror("Error", "The 'players' directory was not found. Please ensure the game files are correctly set up.")
            self.reset_game_controls()
            return
            
        wordlist_path = os.path.join(players_dir, "cm_wordlist.txt")
        if not os.path.exists(wordlist_path):
            self.add_log_entry(f"Error: Wordlist file not found at {wordlist_path}", "debug")
            messagebox.showerror("Error", f"Wordlist file not found at: {wordlist_path}\nPlease ensure the game files are correctly set up.")
            self.reset_game_controls()
            return
        
        # Map selected roles to agent paths
        # All codemaster options map to CodemasterMCTS for now
        # All guesser options map to GuesserEmbeddings for now
        red_cm_path = "human" if red_cm == "Human" else "players.codemasterMCTS.CodemasterMCTS"
        red_g_path = "human" if red_g == "Human" else "players.guesserEMD.GuesserEmbeddings"
        blue_cm_path = "human" if blue_cm == "Human" else "players.codemasterMCTS.CodemasterMCTS"
        blue_g_path = "human" if blue_g == "Human" else "players.guesserEMD.GuesserEmbeddings"
        
        # Build the command
        cmd = ["python", "run_game.py", 
            red_cm_path, red_g_path,
            blue_cm_path, blue_g_path,
            "--seed", seed]
            
        # Initialize the board immediately with loading message
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                tile.config(text="Loading...", bg="#ECF0F1", fg="#2C3E50", relief=tk.RAISED, bd=2)

        # Add game start log entry
        self.add_log_entry("Game Started", "game_event")
        self.add_log_entry("==========================", "separator")
        self.add_log_entry(f"Starting game with command: {' '.join(cmd)}", "debug")
        
        # Start the game in a new thread
        threading.Thread(target=self.run_game, args=(cmd,), daemon=True).start()

    def run_game(self, cmd):
        """Run the game process"""
        try:
            # Redirect stdout for capturing output
            sys.stdout = self.stdout_redirector
            
            # Start the process with modified settings for better output capture
            self.game_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Capture output
            all_output = ""
            board_found = False
            
            for line in iter(self.game_process.stdout.readline, ''):
                #  all output
                all_output += line
                
                self.output_queue.put(line)
                
                if "KEY" in line or "BOARD" in line or "_______" in line:
                    board_section = line
                    for _ in range(40):  # Increased range to capture both sections
                        next_line = self.game_process.stdout.readline()
                        if next_line:
                            all_output += next_line
                            board_section += next_line
                            self.output_queue.put(next_line)
                    
                    self.process_output(board_section)
                    
                    # Check if the board has been initialized yet
                    board_found = any(tile.cget("text") != "Loading..." for row in self.word_tiles for tile in row)
                    if board_found and not self.board_initialized:
                        self.board_initialized = True
                        self.add_log_entry("Board initialized!", "game_event")
                
                if "CLUE:" in line or "clue is" in line:
                    self.process_output(line)
                elif "Guessing:" in line or "guess_answer" in line:
                    self.process_output(line)
                elif "RESULT:" in line:
                    self.process_output(line)
                
                time.sleep(0.01)
            
            # Process is done
            self.game_process.wait()
            
            if not board_found:
                self.add_log_entry("Trying to extract board from entire output...", "debug")
                self.process_output(all_output)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}\n"
            self.output_queue.put(error_msg)
            self.add_log_entry(error_msg, "debug")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        
        finally:
            # Restore stdout
            sys.stdout = self.original_stdout
            
            # Update UI
            self.root.after(0, self.reset_game_controls)

    def reset_game_controls(self):
        """Reset game controls after game completes"""
        self.game_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        
        self.add_log_entry("==========================", "separator")
        self.add_log_entry("Game Ended", "game_event")
    
    def stop_game(self):
        """Stop the running game"""
        if self.game_process and self.game_running:
            self.game_process.terminate()
            
            self.reset_game_controls()

if __name__ == "__main__":
    root = tk.Tk()
    app = CodenamesGUI(root)
    root.mainloop()