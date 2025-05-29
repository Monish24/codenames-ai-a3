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
from datetime import datetime
import csv

try:
    from believability_tournament import BelievabilityTournament
    TOURNAMENT_AVAILABLE = True
except ImportError:
    print("Tournament system not available - tournament features disabled")
    TOURNAMENT_AVAILABLE = False

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
            ("GPT", "GPT-based", False),
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
            ("GPT", "GPT-based", False),
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
        
        # Create window
        self.window = tk.Toplevel(parent.root)
        self.window.title("Tournament Results")
        self.window.geometry("1000x700")
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
                              font=font.Font(family="Helvetica", size=18, weight="bold"),
                              bg="#242731", fg="#E7E1BD")
        title_label.pack(pady=(0, 20))
        
        # Create notebook for different views
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Rankings tab
        rankings_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(rankings_frame, text="Final Rankings")
        
        # Believability tab (if enabled)
        if self.results_data.get('believability_enabled', False):
            believe_frame = tk.Frame(notebook, bg="#242731")
            notebook.add(believe_frame, text="Believability Analysis")
            self.setup_believability_tab(believe_frame)
        
        # Raw data tab
        raw_frame = tk.Frame(notebook, bg="#242731")
        notebook.add(raw_frame, text="Raw Data")
        
        # Setup tabs
        self.setup_rankings_tab(rankings_frame)
        self.setup_raw_data_tab(raw_frame)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg="#242731")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Export button
        export_btn = tk.Button(button_frame, text="Export Results", command=self.export_results,
                              font=font.Font(family="Helvetica", size=12),
                              bg="#FF9800", fg="white", relief=tk.FLAT, bd=0, 
                              padx=20, pady=8)
        export_btn.pack(side=tk.LEFT)
        
        # Close button
        close_btn = tk.Button(button_frame, text="Close", command=self.window.destroy,
                             font=font.Font(family="Helvetica", size=12),
                             bg="#9E9E9E", fg="white", relief=tk.FLAT, bd=0, 
                             padx=20, pady=8)
        close_btn.pack(side=tk.RIGHT)
    
    def setup_rankings_tab(self, parent):
        # Create treeview for rankings
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
        
        # Populate with data
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
    
    def setup_believability_tab(self, parent):
        # Create frame for believability analysis
        believe_frame = tk.Frame(parent, bg="#242731")
        believe_frame.pack(fill=tk.BOTH, expand=True)
        
        # Believability rankings
        believe_label = tk.Label(believe_frame, text="Clue Believability Analysis", 
                                font=font.Font(family="Helvetica", size=14, weight="bold"),
                                bg="#242731", fg="#F1C40F")
        believe_label.pack(pady=(0, 10))
        
        # Create treeview for believability
        b_columns = ("Rank", "Codemaster", "Believability", "Frequency", "Coherence", "Human-like")
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
        if 'believability_data' in self.results_data:
            for i, data in enumerate(self.results_data['believability_data'], 1):
                b_tree.insert("", tk.END, values=(
                    i,
                    data.get('codemaster', 'Unknown'),
                    f"{data.get('overall_believability', 0):.3f}",
                    f"{data.get('frequency_score', 0):.3f}",
                    f"{data.get('semantic_coherence', 0):.3f}",
                    f"{data.get('human_likeness', 0):.3f}"
                ))
    
    def setup_raw_data_tab(self, parent):
        # Create scrollable text area for raw data
        text_frame = tk.Frame(parent, bg="#242731")
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        raw_text = scrolledtext.ScrolledText(text_frame, 
                                           font=font.Font(family="Consolas", size=9),
                                           bg="#1E1F29", fg="#E7E1BD", 
                                           relief=tk.FLAT, bd=1, 
                                           wrap=tk.WORD)
        raw_text.pack(fill=tk.BOTH, expand=True)
        
        # Add raw data
        raw_data = json.dumps(self.results_data, indent=2, default=str)
        raw_text.insert(tk.END, raw_data)
        raw_text.config(state=tk.DISABLED)
    
    def export_results(self):
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
                
                messagebox.showinfo("Export Complete", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")

class CodenamesGUI:
    def __init__(self, root):
        self.guessed_words = set()  
        self.word_results = []  
        self.board_initialized = False  
        self.root = root
        self.root.title("Codenames AI Game & Tournament")
        self.root.geometry("1100x750")
        self.root.configure(bg="#2C3E50")
        
        # Initialize output queue
        self.output_queue = queue.Queue()
        
        self.setup_fonts()
        self.create_widgets()
        self.setup_stdout_redirection()
        
        self.game_process = None
        self.game_running = False
        
        # Tournament variables
        self.tournament_running = False
        self.tournament_thread = None
        self.tournament_progress_window = None
        
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
        title_label = tk.Label(main_frame, text="Codenames AI Game & Tournament", 
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
                                    bg="#4CAF50", fg="black",
                                    relief=tk.FLAT, bd=0, padx=10, width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop game button
        self.stop_button = tk.Button(controls_frame, text="Stop Game", 
                                    command=self.stop_game, 
                                    font=self.button_font,
                                    bg="#F44336", fg="black",
                                    relief=tk.FLAT, bd=0, padx=10, width=15, 
                                    state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Tournament button
        self.tournament_button = tk.Button(controls_frame, text="Tournament", 
                                         command=self.open_tournament_settings, 
                                         font=self.button_font,
                                         bg="#9C27B0", fg="black",
                                         relief=tk.FLAT, bd=0, padx=10, width=15,
                                         state=tk.NORMAL if TOURNAMENT_AVAILABLE else tk.DISABLED)
        self.tournament_button.pack(side=tk.LEFT, padx=5)
        
        # Debug button
        self.debug_button = tk.Button(controls_frame, text="Show Debug Info", 
                                    command=self.show_debug_info, 
                                    font=self.button_font,
                                    bg="#FF9800", fg="black",
                                    relief=tk.FLAT, bd=0, padx=10, width=15)
        self.debug_button.pack(side=tk.LEFT, padx=5)
        
        # Spymaster toggle button
        self.spymaster_button = tk.Button(controls_frame, text="Spymaster View", 
                                        command=self.toggle_spymaster_view, 
                                        font=self.button_font,
                                        bg="#673AB7", fg="black",
                                        relief=tk.FLAT, bd=0, padx=10, width=18)
        self.spymaster_button.pack(side=tk.LEFT, padx=5)
        
        # Game configuration frame
        config_frame = tk.Frame(main_frame, bg="#242731")
        config_frame.pack(fill=tk.X, pady=5)
        
        # Team configuration section - 4 dropdowns
        role_options = ["Human", "MCTS", "EMD", "GPT", "SBERT", "CL", "TOT", "Naive"]
        
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

        self.red_guesser = tk.StringVar(value="EMD")
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

        self.blue_codemaster = tk.StringVar(value="EMD")
        blue_cm_dropdown = tk.OptionMenu(blue_cm_frame, self.blue_codemaster, *role_options)
        blue_cm_dropdown.config(font=self.board_font, bg="#34495E", fg="white", width=12)
        blue_cm_dropdown.pack(side=tk.LEFT, padx=2)

        # Blue Guesser dropdown
        blue_g_frame = tk.Frame(blue_team_frame, bg="#242731")
        blue_g_frame.pack(fill=tk.X, pady=2)

        blue_g_label = tk.Label(blue_g_frame, text="Guesser:", 
                            font=self.board_font, bg="#242731", fg="#E7E1BD")
        blue_g_label.pack(side=tk.LEFT, padx=2)

        self.blue_guesser = tk.StringVar(value="EMD")
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
        
        # Content pane - split into board and game info
        content_paned = tk.PanedWindow(main_frame, bg="#242731", orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Game board frame (left side)
        board_frame = tk.Frame(content_paned, bg="#1E1F29", bd=0)
        board_frame.configure(highlightbackground="#1A1A1A", highlightthickness=1)
        
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
        
        content_paned.add(board_frame, stretch="always", width=600)
        content_paned.add(info_frame, stretch="always", width=400)
        
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
        self.log_text.tag_configure("tournament", foreground="#9C27B0",
                                    font=font.Font(family="Helvetica", size=12, weight="bold"))
        
        self.setup_button_hover()
        
        # Show tournament availability
        if not TOURNAMENT_AVAILABLE:
            self.add_log_entry("Tournament system not available - check imports", "debug")
    
    def setup_button_hover(self):
        """Setup hover effects for buttons"""
        def on_enter(e, button, hover_color):
            button['background'] = hover_color
            
        def on_leave(e, button, original_color):
            button['background'] = original_color
        
        # Set up hover effects for each button
        buttons = [
            (self.start_button, "#4CAF50", "#81C784"),
            (self.stop_button, "#F44336", "#E57373"),
            (self.tournament_button, "#9C27B0", "#BA68C8"),
            (self.debug_button, "#FF9800", "#FFB74D"),
            (self.spymaster_button, "#673AB7", "#9575CD")
        ]
        
        for button, original, hover in buttons:
            button.bind("<Enter>", lambda e, b=button, h=hover: on_enter(e, b, h))
            button.bind("<Leave>", lambda e, b=button, o=original: on_leave(e, b, o))

    def get_tournament_clue_data(self):
        """Return collected clue data for believability analysis"""
        if hasattr(self, 'tournament_clues'):
            return self.tournament_clues.copy()
        return []

    def clear_tournament_clue_data(self):
        """Clear stored clue data (call at start of new tournament)"""
        self.tournament_clues = []

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
                self.process_output(output)  # Use process_output directly
        except Exception as e:
            self.add_log_entry(f"Error processing output: {str(e)}", "debug")
        
        # Schedule the next check
        self.root.after(100, self.check_output_queue)
    
    def process_output(self, output):
        """Process captured output - COMPLETELY REWRITTEN for sequential flow"""
        
        # Skip empty output
        if not output or not output.strip():
            return
        
        # Clean ANSI color codes
        clean_output = re.sub(r'\x1b\[[0-9;]*m', '', output).strip()
        
        # Initialize tracking if needed
        if not hasattr(self, '_game_state'):
            self._game_state = {
                'board_extracted': False,
                'current_turn': None,
                'current_clue': None,
                'current_clue_num': 0,
                'expecting_guess': False,
                'expecting_result': False,
                'last_guess': None,
                'processed_lines': set()
            }
        
        # Skip if we've seen this exact line before
        line_hash = hash(clean_output)
        if line_hash in self._game_state['processed_lines']:
            return
        self._game_state['processed_lines'].add(line_hash)
        
        # STEP 1: BOARD EXTRACTION (only once)
        if not self._game_state['board_extracted'] and ("KEY" in clean_output or "BOARD" in clean_output):
            if self.try_extract_board(clean_output):
                self._game_state['board_extracted'] = True
                self.add_log_entry("Game Started", "game_event")
                return
        
        # STEP 2: TEAM TURN DETECTION
        if "RED TEAM TURN" in clean_output:
            self._game_state['current_turn'] = "Red"
            self._game_state['expecting_guess'] = False
            self._game_state['expecting_result'] = False
            self.current_turn = "Red"
            self.turn_label.config(text="Red Team's Turn", fg="#C13A37")
            self.add_log_entry("Red Team's Turn", "red_turn")
            return
        
        if "BLUE TEAM TURN" in clean_output:
            self._game_state['current_turn'] = "Blue"
            self._game_state['expecting_guess'] = False
            self._game_state['expecting_result'] = False
            self.current_turn = "Blue"
            self.turn_label.config(text="Blue Team's Turn", fg="#4989C5")
            self.add_log_entry("Blue Team's Turn", "blue_turn")
            return
        
        # STEP 3: CLUE DETECTION
        clue_patterns = [
            r"STRUCTURED_CLUE:\s*([^|]+)\|([^|]+)\|(\d+)\|([^|]+)",
            r"The clue is:\s*(\w+)\s+(\d+)",
            r"clue is:\s*(\w+)\s+(\d+)"
        ]
        
        for pattern in clue_patterns:
            match = re.search(pattern, clean_output, re.IGNORECASE)
            if match:
                if "STRUCTURED_CLUE" in pattern:
                    _, clue_word, clue_num, team = match.groups()
                else:
                    clue_word, clue_num = match.groups()
                    team = self._game_state['current_turn']
                
                clue_word = clue_word.strip().upper()
                clue_num = int(clue_num)
                
                # Only process if this is a new clue
                if self._game_state['current_clue'] != clue_word or self._game_state['current_clue_num'] != clue_num:
                    self._game_state['current_clue'] = clue_word
                    self._game_state['current_clue_num'] = clue_num
                    self._game_state['expecting_guess'] = True
                    
                    # Update display
                    clue_text = f"{clue_word} ({clue_num})"
                    if "RED" in team.upper() or self._game_state['current_turn'] == "Red":
                        self.clue_label.config(text=clue_text, fg="#E74C3C")
                        self.add_log_entry(f"{self._game_state['current_turn']} Codemaster Gave Clue: {clue_word} {clue_num}", "red_clue")
                    else:
                        self.clue_label.config(text=clue_text, fg="#3498DB")
                        self.add_log_entry(f"{self._game_state['current_turn']} Codemaster Gave Clue: {clue_word} {clue_num}", "blue_clue")
                return
        
        # STEP 4: GUESS DETECTION
        if self._game_state['expecting_guess']:
            guess_patterns = [
                r"Guessing:\s*([A-Z]+)",
                r"Selected:\s*([A-Z]+)",
                r"Guesser selected:\s*([A-Z]+)"
            ]
            
            for pattern in guess_patterns:
                match = re.search(pattern, clean_output, re.IGNORECASE)
                if match:
                    guess_word = match.group(1).upper()
                    
                    # Only process if this is a new guess
                    if self._game_state['last_guess'] != guess_word:
                        self._game_state['last_guess'] = guess_word
                        self._game_state['expecting_guess'] = False
                        self._game_state['expecting_result'] = True
                        
                        # Update display
                        self.guess_label.config(text=f"{guess_word} ⏳")
                        team_color = "red_guess" if self._game_state['current_turn'] == "Red" else "blue_guess"
                        self.add_log_entry(f"{self._game_state['current_turn']} Guesser Guessed: {guess_word}", team_color)
                    return
        
        # STEP 5: RESULT DETECTION (only when expecting one)
        if self._game_state['expecting_result'] and self._game_state['last_guess']:
            # Look for results in a very specific way
            word = self._game_state['last_guess']
            
            # Find what team this word belongs to from our board data
            word_team = None
            for result_word, team in self.word_results:
                if result_word.upper() == word.upper():
                    word_team = team
                    break
            
            if word_team:
                self._game_state['expecting_result'] = False
                self.process_single_guess_result(word, word_team)
                
                # Check if we should expect another guess
                if self._game_state['current_clue_num'] > 1:
                    self._game_state['expecting_guess'] = True
                    self._game_state['current_clue_num'] -= 1
                else:
                    self._game_state['expecting_guess'] = False
                return
        
        # STEP 6: WIN DETECTION
        if "Red Team Wins" in clean_output or "RED WINS" in clean_output:
            self.add_log_entry("🏆 RED TEAM WINS! 🏆", "win")
            self.turn_label.config(text="🏆 Red Team Wins! 🏆", fg="#C13A37")
            self.add_log_entry("Game Ended", "game_event")
            return
        
        if "Blue Team Wins" in clean_output or "BLUE WINS" in clean_output:
            self.add_log_entry("🏆 BLUE TEAM WINS! 🏆", "win")
            self.turn_label.config(text="🏆 Blue Team Wins! 🏆", fg="#4989C5")
            self.add_log_entry("Game Ended", "game_event")
            return
        
        # STEP 7: SEED DETECTION
        if "seed:" in clean_output.lower():
            self.add_log_entry(f"ℹ️ {clean_output}", "debug")
            return
    def reset_game_state(self):
        """Reset game state for new game"""
        # Reset tracking variables
        self.guessed_words = set()
        self.word_results = []
        self.board_initialized = False
        
        # Reset statistics
        self.red_words_found = 0
        self.blue_words_found = 0
        self.civilian_words_found = 0
        self.assassin_found = False
        
        # Reset display
        self.current_turn = "Red"
        self.clue_label.config(text="Waiting for clue...")
        self.guess_label.config(text="Waiting for guess...")
        self.red_score.config(text="0/9")
        self.blue_score.config(text="0/8")
        self.turn_label.config(text="")
        
        # Reset game state tracking
        if hasattr(self, '_game_state'):
            del self._game_state
        
        # Reset spymaster mode to default
        self.spymaster_mode = True
        
        # Reset all tiles
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                tile.config(text="Loading...", bg="#E7E1BD", fg="#2C3E50", relief=tk.RAISED, bd=2)

    def process_single_guess_result(self, word, team_type):
        """Process a single guess result in the proper sequence"""
        word = word.upper()
        team_type = team_type.lower()
        
        current_turn = self._game_state['current_turn']
        
        # Determine if this was correct for the current team
        if team_type == "red":
            if current_turn == "Red":
                emoji = "✅"
                result_text = f"Correct: {word} is a Red Card"
                self.red_words_found += 1
                self.red_score.config(text=f"{self.red_words_found}/9")
                team_color = "red_guess"
            else:
                emoji = "❌"
                result_text = f"Wrong: {word} is a Red Card (Turn Ends)"
                team_color = "blue_guess"
                
        elif team_type == "blue":
            if current_turn == "Blue":
                emoji = "✅"
                result_text = f"Correct: {word} is a Blue Card"
                self.blue_words_found += 1
                self.blue_score.config(text=f"{self.blue_words_found}/8")
                team_color = "blue_guess"
            else:
                emoji = "❌"
                result_text = f"Wrong: {word} is a Blue Card (Turn Ends)"
                team_color = "red_guess"
                
        elif team_type == "civilian":
            emoji = "⚪"
            result_text = f"Neutral: {word} is a Civilian Card (Turn Ends)"
            team_color = "game_event"
            self.civilian_words_found += 1
            
        elif team_type == "assassin":
            emoji = "☠️"
            result_text = f"ASSASSIN: {word} is the Assassin! (Game Over)"
            team_color = "win"
            self.assassin_found = True
        
        else:
            emoji = "❓"
            result_text = f"Unknown: {word}"
            team_color = "game_event"
        
        # Update display
        self.guess_label.config(text=f"{word} {emoji}")
        self.add_log_entry(result_text, team_color)
        
        # Update the tile
        self.update_tile_for_word(word)
        
        # Add to guessed words
        self.guessed_words.add(word)
        
        print(f"DEBUG: Processed result: {result_text}")

    def extract_board_from_display(self, output):
        """Extract board from the game output with REAL key grid and board data"""
        
        # PREVENT DUPLICATE EXTRACTIONS
        if self.board_initialized:
            print("DEBUG: Board already initialized, skipping fallback extraction")
            return True
        
        lines = output.split('\n')
        
        # Look for KEY section and BOARD section
        key_teams = []
        board_words = []
        
        print(f"DEBUG: Processing {len(lines)} lines of output")
        
        # Find KEY section
        key_start = -1
        board_start = -1
        
        for i, line in enumerate(lines):
            if "KEY" in line and "_" in line:
                key_start = i
                print(f"DEBUG: Found KEY section at line {i}")
            elif "BOARD" in line and "_" in line:
                board_start = i
                print(f"DEBUG: Found BOARD section at line {i}")
        
        # Extract key grid (team assignments)
        if key_start >= 0:
            for i in range(key_start + 1, min(key_start + 10, len(lines))):
                line = lines[i].strip()
                if not line or "_" in line:
                    break
                
                # Remove ANSI color codes
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                
                # Extract team words from the line
                teams_in_line = []
                words = clean_line.split()
                for word in words:
                    word = word.strip()
                    if word in ['Red', 'Blue', 'Civilian', 'Assassin']:
                        teams_in_line.append(word.lower())
                
                key_teams.extend(teams_in_line)
                if teams_in_line:  # Only print if we found teams
                    print(f"DEBUG: Key line: '{clean_line}' -> {teams_in_line}")
        
        # Extract board words
        if board_start >= 0:
            for i in range(board_start + 1, min(board_start + 10, len(lines))):
                line = lines[i].strip()
                if not line or "_" in line:
                    break
                
                # Remove ANSI color codes
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                
                # Extract words (look for capitalized words that aren't team names)
                words = re.findall(r'\b[A-Z]{2,}\b', clean_line)
                words_filtered = [w for w in words if w not in ['RED', 'BLUE', 'CIVILIAN', 'ASSASSIN']]
                
                board_words.extend(words_filtered)
                if words_filtered:  # Only print if we found words
                    print(f"DEBUG: Board line: '{clean_line}' -> {words_filtered}")
        
        print(f"DEBUG: Extracted {len(key_teams)} key assignments and {len(board_words)} words")
        
        # Create word-team mappings if we have both
        if len(key_teams) >= 25 and len(board_words) >= 25:
            # CLEAR any existing data first
            self.word_results = []
            
            # Create exactly 25 mappings
            for i in range(25):
                if i < len(key_teams) and i < len(board_words):
                    self.word_results.append((board_words[i], key_teams[i]))
            
            print(f"DEBUG: Created {len(self.word_results)} word-team mappings")
            for i, (word, team) in enumerate(self.word_results[:5]):
                print(f"DEBUG: {i}: {word} -> {team}")
            
            self.initialize_board_with_words(board_words[:25])
            return True
        else:
            print("DEBUG: Not enough data for extraction")
        
        return False

    def try_extract_board(self, output):
        """Improved board extraction that handles the actual game.py output format"""
        
        # PREVENT DUPLICATE EXTRACTIONS - only extract once per game
        if self.board_initialized:
            print("DEBUG: Board already initialized, skipping extraction")
            return True
        
        # Method 1: Look for the complete KEY + BOARD pattern from game.py
        key_pattern = r"____________________________KEY____________________________"
        board_pattern = r"___________________________BOARD___________________________"
        
        if key_pattern in output and board_pattern in output:
            key_start = output.find(key_pattern)
            board_start = output.find(board_pattern)
            
            if key_start >= 0 and board_start >= 0:
                # Extract key section
                key_section = output[key_start:board_start]
                board_section = output[board_start:board_start + 1000]  # Get reasonable amount after board
                
                # Parse key section
                key_teams = []
                key_lines = key_section.split('\n')
                for line in key_lines:
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    # Look for team names in order they appear
                    teams = re.findall(r'\b(Red|Blue|Civilian|Assassin)\b', clean_line)
                    key_teams.extend([t.lower() for t in teams])
                
                # Parse board section  
                board_words = []
                board_lines = board_section.split('\n')
                for line in board_lines:
                    clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    # Extract actual word names (not team names)
                    words = re.findall(r'\b[A-Z]{2,}\b', clean_line)
                    filtered_words = [w for w in words if w not in ['RED', 'BLUE', 'CIVILIAN', 'ASSASSIN', 'KEY', 'BOARD']]
                    board_words.extend(filtered_words)
                
                print(f"DEBUG: Pattern method found {len(key_teams)} teams, {len(board_words)} words")
                
                # ONLY proceed if we have EXACTLY the right amount of data
                if len(key_teams) >= 25 and len(board_words) >= 25:
                    # CLEAR any existing data first
                    self.word_results = []
                    
                    # Create mappings for exactly 25 words
                    for i in range(25):
                        if i < len(key_teams) and i < len(board_words):
                            self.word_results.append((board_words[i], key_teams[i]))
                            print(f"DEBUG: Mapping {board_words[i]} -> {key_teams[i]}")
                    
                    print(f"DEBUG: Final word_results has {len(self.word_results)} entries")
                    
                    # Initialize board with words
                    self.initialize_board_with_words(board_words[:25])
                    return True
        
        # Method 2: Try the fallback extraction method
        return self.extract_board_from_display(output)

    def initialize_board_with_words(self, words):
        """Initialize the board with actual words - ONLY ONCE"""
        
        # PREVENT MULTIPLE INITIALIZATIONS
        if self.board_initialized:
            print("DEBUG: Board already initialized, skipping")
            return
        
        # Ensure we have exactly 25 words
        while len(words) < 25:
            words.append(f"WORD{len(words)+1}")
        
        words = words[:25]
        
        # Store the board words
        self.board_words = words
        
        # Update board tiles with words
        for i in range(5):
            for j in range(5):
                idx = i*5 + j
                word = words[idx]
                tile = self.word_tiles[i][j]
                tile.config(text=word, bg="#E7E1BD", fg="black", relief=tk.RAISED, bd=2)
        
        # Mark as initialized FIRST to prevent re-initialization
        self.board_initialized = True
        
        # Update button text to match current mode
        if self.spymaster_mode:
            self.spymaster_button.config(text="Player View")
        else:
            self.spymaster_button.config(text="Spymaster View")
        
        print(f"DEBUG: Board initialized with {len(self.word_results)} team mappings")
        
        # Apply the current view ONCE
        self.update_board_display()
        
        self.add_log_entry("📋 Board initialized!", "game_event")

    def update_board_display(self):
        """Update the board display based on current spymaster mode"""
        if not self.board_initialized:
            print("DEBUG: Board not initialized, skipping display update")
            return
        
        # Throttle updates - don't update more than once per 100ms
        current_time = time.time()
        if hasattr(self, '_last_display_update'):
            if current_time - self._last_display_update < 0.1:
                return
        self._last_display_update = current_time
        
        # Debug log (reduced)
        print(f"DEBUG: Updating board display, spymaster_mode={self.spymaster_mode}, word_results={len(self.word_results)}")
        
        # Define colors
        red_color = "#C13A37"      
        blue_color = "#4989C5"     
        civilian_color = "#DCD6B0" 
        assassin_color = "#2C2C2E" 
        neutral_color = "#E7E1BD"  
        
        # Update all tiles based on current knowledge and mode
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                word = tile.cget("text")
                
                if not word or word == "Loading...":
                    continue
                
                # Check if this word has been guessed
                is_guessed = word.upper() in self.guessed_words
                
                # Find the team for this word - use FIRST match only
                team = "unknown"
                word_upper = word.upper()
                
                for result_word, result_team in self.word_results:
                    if result_word.upper() == word_upper:
                        team = result_team.lower()
                        break  # Use FIRST match only
                
                # Set the appropriate style based on team and current view mode
                if self.spymaster_mode:
                    # Spymaster view - show all team colors regardless of guessed status
                    if team == "red":
                        tile.config(bg=red_color, fg="white")
                    elif team == "blue":
                        tile.config(bg=blue_color, fg="white")
                    elif team == "civilian":
                        tile.config(bg=civilian_color, fg="black")
                    elif team == "assassin":
                        tile.config(bg=assassin_color, fg="white")
                    else:
                        # Unknown team - use neutral color
                        tile.config(bg=neutral_color, fg="black")
                    
                    # Set relief based on guessed status in spymaster view too
                    if is_guessed:
                        tile.config(relief=tk.SUNKEN, bd=3)
                    else:
                        tile.config(relief=tk.RAISED, bd=2)
                        
                else:
                    # Player view - only show team colors for guessed cards
                    if is_guessed:
                        # Revealed cards show their team color
                        if team == "red":
                            tile.config(bg=red_color, fg="white", relief=tk.SUNKEN, bd=3)
                        elif team == "blue":
                            tile.config(bg=blue_color, fg="white", relief=tk.SUNKEN, bd=3)
                        elif team == "civilian":
                            tile.config(bg=civilian_color, fg="black", relief=tk.SUNKEN, bd=3)
                        elif team == "assassin":
                            tile.config(bg=assassin_color, fg="white", relief=tk.SUNKEN, bd=3)
                        else:
                            tile.config(bg="#999999", fg="white", relief=tk.SUNKEN, bd=3)  # Unknown guessed
                    else:
                        # Unrevealed cards look like neutral cards
                        tile.config(bg=neutral_color, fg="black", relief=tk.RAISED, bd=2)

    def toggle_spymaster_view(self):
        """Toggle between player view and spymaster view"""
        if not self.board_initialized:
            self.add_log_entry("⚠️ Board not initialized yet!", "debug")
            return
        
        # Toggle the mode
        self.spymaster_mode = not self.spymaster_mode
        
        # Debug log
        print(f"DEBUG: Toggled to spymaster_mode={self.spymaster_mode}")
        
        # Update the button text
        if self.spymaster_mode:
            self.spymaster_button.config(text="Player View")
            self.add_log_entry("👁️ Switched to Spymaster View", "debug")
        else:
            self.spymaster_button.config(text="Spymaster View")
            self.add_log_entry("🕶️ Switched to Player View", "debug")
        
        # Update the board display based on the new mode
        self.update_board_display()
    def update_clue_with_team(self, clue_word, clue_num, team):
        """Update clue with specific team information"""
        self.current_clue = clue_word.upper()
        self.current_clue_num = int(clue_num)
        
        clue_text = f"{self.current_clue} ({self.current_clue_num})"
        
        # Determine team from the team parameter
        if "RED" in team.upper():
            self.current_turn = "Red"
            self.clue_label.config(text=clue_text, fg="#E74C3C")
            self.add_log_entry(f"💭 Red Clue: {clue_text}", "red_clue")
        elif "BLUE" in team.upper():
            self.current_turn = "Blue"
            self.clue_label.config(text=clue_text, fg="#3498DB")
            self.add_log_entry(f"💭 Blue Clue: {clue_text}", "blue_clue")
        else:
            # Fallback to current turn
            self.update_clue(clue_word, clue_num)

    def update_clue(self, clue_word, clue_num):
        """Update the current clue display and log (improved)"""
        self.current_clue = clue_word.upper()
        self.current_clue_num = int(clue_num)
        
        clue_text = f"{self.current_clue} ({self.current_clue_num})"
        
        if self.current_turn == "Red":
            self.clue_label.config(text=clue_text, fg="#E74C3C")
            self.add_log_entry(f"💭 Clue: {clue_text}", "red_clue")
        else:
            self.clue_label.config(text=clue_text, fg="#3498DB")
            self.add_log_entry(f"💭 Clue: {clue_text}", "blue_clue")

    def update_guess(self, guess_word):
        """Update the current guess display and log - IMPROVED"""
        self.current_guess = guess_word.upper()
        
        # Show pending guess state
        self.guess_label.config(text=f"{self.current_guess} ⏳")
        
        if self.current_turn == "Red":
            self.add_log_entry(f"🤔 Guessing: {self.current_guess}", "red_guess")
        else:
            self.add_log_entry(f"🤔 Guessing: {self.current_guess}", "blue_guess")

    def process_guess_result(self, word, team_type):
        """Process the result of a guess - IMPROVED"""
        word = word.upper()
        team_type = team_type.lower()
        
        print(f"DEBUG: Processing guess result: {word} -> {team_type}")
        
        # Determine emoji and outcome
        if team_type == "red":
            emoji = "✅" if self.current_turn == "Red" else "❌"
            team_color = "red_guess" if self.current_turn == "Red" else "blue_guess"
            self.word_results.append((word, "red"))
            
            if self.current_turn == "Red":
                self.red_words_found += 1
                self.red_score.config(text=f"{self.red_words_found}/9")
                
        elif team_type == "blue":
            emoji = "✅" if self.current_turn == "Blue" else "❌"
            team_color = "blue_guess" if self.current_turn == "Blue" else "red_guess"
            self.word_results.append((word, "blue"))
            
            if self.current_turn == "Blue":
                self.blue_words_found += 1
                self.blue_score.config(text=f"{self.blue_words_found}/8")
                
        elif team_type == "civilian":
            emoji = "⚪"
            team_color = "game_event"
            self.word_results.append((word, "civilian"))
            self.civilian_words_found += 1
            
        elif team_type == "assassin":
            emoji = "☠️"
            team_color = "win"
            self.word_results.append((word, "assassin"))
            self.assassin_found = True
        
        else:
            emoji = "❓"
            team_color = "game_event"
        
        # Add to log with proper formatting
        result_text = f"{emoji} {word} → {team_type.title()}"
        self.add_log_entry(result_text, team_color)
        
        # Update the tile
        self.update_tile_for_word(word)
        
        # Update guess display with result
        self.guess_label.config(text=f"{word} {emoji}")
        
        print(f"DEBUG: Result processed: {result_text}")

    def update_tile_for_word(self, word):
        """Update the tile for a specific word"""
        word = word.upper()
        
        # Find the team for this word
        team = "unknown"
        for result_word, result_team in self.word_results:
            if result_word.upper() == word:
                team = result_team.lower()
                break
        
        # Flag to track if this is the first time revealing this word
        first_reveal = word not in self.guessed_words
        
        # Add to guessed words
        self.guessed_words.add(word)
        
        # Update the tile appearance
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                if tile.cget("text").upper() == word:
                    # Always update the tile to show it's been guessed
                    if team == "red":
                        tile.config(bg="#C13A37", fg="white", relief=tk.SUNKEN, bd=3)
                    elif team == "blue":
                        tile.config(bg="#4989C5", fg="white", relief=tk.SUNKEN, bd=3)
                    elif team == "civilian":
                        tile.config(bg="#DCD6B0", fg="black", relief=tk.SUNKEN, bd=3)
                    elif team == "assassin":
                        tile.config(bg="#2C2C2E", fg="white", relief=tk.SUNKEN, bd=3)
                    else:
                        # Unknown team - generic guessed appearance
                        tile.config(bg="#999999", fg="white", relief=tk.SUNKEN, bd=3)
                    break
        
        # If we're in player view, make sure the display updates correctly
        if not self.spymaster_mode:
            self.update_board_display()

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

    def reset_board_for_new_game(self):
        """Reset board state for a new tournament game"""
        # Reset tracking variables
        self.guessed_words = set()
        self.word_results = []
        self.board_initialized = False
        
        # Reset statistics
        self.red_words_found = 0
        self.blue_words_found = 0
        self.civilian_words_found = 0
        self.assassin_found = False
        
        # Reset display
        self.current_turn = "Red"
        self.clue_label.config(text="Waiting for clue...")
        self.guess_label.config(text="Waiting for guess...")
        self.red_score.config(text="0/9")
        self.blue_score.config(text="0/8")
        self.turn_label.config(text="")
        
        # Reset spymaster mode to default but don't change button text until board is initialized
        self.spymaster_mode = True
        
        # Reset all tiles
        for i in range(5):
            for j in range(5):
                tile = self.word_tiles[i][j]
                tile.config(text="Loading...", bg="#E7E1BD", fg="#2C3E50", relief=tk.RAISED, bd=2)

    def show_debug_info(self):
        """Show debug information about the game state"""
        debug_info = [
            "=== DEBUG INFORMATION ===",
            f"Current directory: {os.getcwd()}",
            f"Game running: {self.game_running}",
            f"Tournament available: {TOURNAMENT_AVAILABLE}",
            f"Current turn: {self.current_turn}",
            f"Clue: {self.current_clue} ({self.current_clue_num})",
            f"Guess: {self.current_guess}",
            f"Red words found: {self.red_words_found}/9",
            f"Blue words found: {self.blue_words_found}/8",
            f"Civilian words found: {self.civilian_words_found}",
            f"Assassin found: {self.assassin_found}",
            f"Board initialized: {self.board_initialized}",
            f"Spymaster mode: {self.spymaster_mode}",
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
                    'guesser': None
                },
                'TOT': {
                    'codemaster': ('players.codemaster_TOT', 'CodemasterTreeOfThoughts'),
                    'guesser': None
                },
                'Naive': {
                    'codemaster': None,
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
                        # Update progress window
                        if self.tournament_progress_window and not self.tournament_progress_window.cancelled:
                            self.root.after(0, lambda: self.tournament_progress_window.update_progress(
                                games_completed, total_games, f"Game {games_completed} completed"))
                            
                            # Add log entry for recent matches
                            if tournament.match_results:
                                recent_match = tournament.match_results[-1]
                                match_desc = f"{recent_match.red_codemaster}+{recent_match.red_guesser} vs {recent_match.blue_codemaster}+{recent_match.blue_guesser}"
                                self.root.after(0, lambda: self.tournament_progress_window.add_log_entry(
                                    f"Match {games_completed}: {match_desc} -> {recent_match.winner} wins"))
                        
                        # Check for cancellation
                        if self.tournament_progress_window and self.tournament_progress_window.cancelled:
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
            
            # Prepare results data
            results_data = {
                'tournament_name': tournament.tournament_name,
                'total_games': len(tournament.match_results),
                'believability_enabled': config['believability_analysis'],
                'rankings': []
            }
            
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
            
            if config['believability_analysis']:
                results_data['believability_data'] = []
            
            # Show results
            self.root.after(0, lambda: self.show_tournament_results(results_data))
            
        except Exception as e:
            error_msg = f"Tournament error: {str(e)}"
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

    def start_game(self):
        """Start the game with the selected configuration"""
        if self.game_running:
            return
        
        self.game_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Call the new reset method
        self.reset_game_state()
        
        # Clear the log
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
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
        
        def get_agent_path(agent_type, role_type):
            """Map dropdown selection to actual agent path"""
            if agent_type == "Human":
                return "human"
            
            # Codemaster mappings
            codemaster_map = {
                "MCTS": "players.codemasterMCTS.CodemasterMCTS",
                "EMD": "players.codemaster_EMD.CodemasterEmbeddings", 
                "GPT": "players.codemaster_GPT.CodemasterGPT",
                "SBERT": "players.codemaster_SBERT.CodemasterSBERT",
                "CL": "players.codemaster_CL.CodemasterCurriculum",
                "TOT": "players.codemaster_TOT.CodemasterTreeOfThoughts",
                "Naive": "players.codemasterMCTS.CodemasterMCTS"
            }
            
            # Guesser mappings  
            guesser_map = {
                "MCTS": "players.guesser_MCTS.GuesserMCTS",
                "EMD": "players.guesserEMD.GuesserEmbeddings",
                "GPT": "players.guesser_GPT.GuesserGPT", 
                "SBERT": "players.guesser_SBERT.GuesserSBERT",
                "CL": "players.guesserEMD.GuesserEmbeddings",
                "TOT": "players.guesserEMD.GuesserEmbeddings",
                "Naive": "players.guesser_naive.NaiveGuesser"
            }
            
            if role_type == "codemaster":
                return codemaster_map.get(agent_type, "players.codemasterMCTS.CodemasterMCTS")
            else:
                return guesser_map.get(agent_type, "players.guesserEMD.GuesserEmbeddings")

        # Map selected roles to agent paths
        red_cm_path = get_agent_path(red_cm, "codemaster")
        red_g_path = get_agent_path(red_g, "guesser") 
        blue_cm_path = get_agent_path(blue_cm, "codemaster")
        blue_g_path = get_agent_path(blue_g, "guesser")
        
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
                all_output += line
                
                self.output_queue.put(line)
                
                if "KEY" in line or "BOARD" in line or "_______" in line:
                    board_section = line
                    for _ in range(40):
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