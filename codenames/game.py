import random
import time
import json
import enum
import os
import shutil
import sys
import colorama
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


class GameCondition(enum.Enum):
    """Enumeration that represents the different states of the game"""
    RED_TURN = 0
    BLUE_TURN = 1
    RED_WIN = 2
    BLUE_WIN = 3


@dataclass
class ClueRecord:
    """Record of a clue given during the game"""
    clue_word: str
    clue_number: int
    codemaster_name: str
    target_words: List[str]  # Intended targets (from key grid)
    guessed_words: List[str]  # What was actually guessed
    correct_guesses: List[str]  # Correct guesses from this clue
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


class Game:
    """Enhanced Game class with comprehensive data collection
    """

    def __init__(self, codemaster_red, guesser_red, codemaster_blue, guesser_blue,
                 seed="time", do_print=True, do_log=True, game_name="default",
                 cmr_kwargs={}, gr_kwargs={}, cmb_kwargs={}, gb_kwargs={},
                 single_team=False, collect_data=True):
        """ Setup Game details with enhanced data collection

        Args:
            codemaster_red (:class:`Codemaster`):
                Codemaster for red team (spymaster in Codenames' rules) class that provides a clue.
            guesser_red (:class:`Guesser`):
                Guesser for red team (field operative in Codenames' rules) class that guesses based on clue.
            codemaster_blue (:class:`Codemaster`):
                Codemaster for blue team (spymaster in Codenames' rules) class that provides a clue.
            guesser_blue (:class:`Guesser`):
                Guesser for blue team (field operative in Codenames' rules) class that guesses based on clue.
            seed (int or str, optional): 
                Value used to init random, "time" for time.time(). 
                Defaults to "time".
            do_print (bool, optional): 
                Whether to keep on sys.stdout or turn off. 
                Defaults to True.
            do_log (bool, optional): 
                Whether to append to log file or not. 
                Defaults to True.
            game_name (str, optional): 
                game name used in log file. Defaults to "default".
            cmr_kwargs (dict, optional):
                kwargs passed to red Codemaster.
            gr_kwargs (dict, optional):
                kwargs passed to red Guesser.
            cmb_kwargs (dict, optional):
                kwargs passed to blue Codemaster.
            gb_kwargs (dict, optional):
                kwargs passed to blue Guesser.
            single_team (bool, optional): 
                Whether to play the single team track version. 
                Defaults to False.
            collect_data (bool, optional):
                Whether to collect detailed clue and guess data.
                Defaults to True.
        """

        self.game_winner = None
        self.game_start_time = time.time()
        self.game_end_time = None
        colorama.init()

        self.do_print = do_print
        if not self.do_print:
            self._save_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        self.codemaster_red = codemaster_red("Red", **cmr_kwargs)
        self.guesser_red = guesser_red("Red", **gr_kwargs)
        self.codemaster_blue = codemaster_blue("Blue", **cmb_kwargs)
        self.guesser_blue = guesser_blue("Blue", **gb_kwargs)

        self.cmr_kwargs = cmr_kwargs
        self.gr_kwargs = gr_kwargs
        self.cmb_kwargs = cmb_kwargs
        self.gb_kwargs = gb_kwargs
        self.do_log = do_log
        self.game_name = game_name
        self.single_team = single_team
        self.collect_data = collect_data

        self.num_red_words = 9
        self.num_blue_words = 8
        self.num_civilian_words = 7
        self.num_assassin_words = 1

        # Enhanced data collection
        self.clue_records: List[ClueRecord] = []
        self.guess_records: List[GuessRecord] = []
        self.turn_counter = 0
        self.current_clue = None
        self.current_clue_number = 0
        self.current_codemaster = None
        self.current_guesser = None
        self.current_team_color = None
        self.guess_order_in_turn = 0
        
        # Performance tracking
        self.red_turns_taken = 0
        self.blue_turns_taken = 0
        self.cards_revealed_per_turn = []
        self.turn_start_times = []

        # set seed so that board/keygrid can be reloaded later
        if seed == 'time':
            self.seed = time.time()
            random.seed(self.seed)
        else:
            self.seed = seed
            random.seed(int(seed))

        print("seed:", self.seed)
        sys.stdout.flush()

        # load board words
        wordlist_path = os.path.join(os.path.dirname(__file__), "players", "cm_wordlist.txt")
        with open(wordlist_path, "r") as f:
            temp = f.read().splitlines()
            assert len(temp) == len(set(temp)), "game wordpool should not have duplicates"
            random.shuffle(temp)
            self.words_on_board = temp[:25]

        # set grid key for codemaster (spymaster)
        self.key_grid = ["Red"] * self.num_red_words + ["Blue"] * self.num_blue_words + \
                        ["Civilian"] * self.num_civilian_words + ["Assassin"] * self.num_assassin_words
        random.shuffle(self.key_grid)

    def __del__(self):
        """reset stdout if using the do_print==False option"""
        if not self.do_print:
            sys.stdout.close()
            sys.stdout = self._save_stdout

    def _display_board_codemaster(self):
        """prints out board with color-paired words, only for codemaster, color && stylistic"""
        print(str.center("___________________________BOARD___________________________\n", 60))
        counter = 0
        for i in range(len(self.words_on_board)):
            if counter >= 1 and i % 5 == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RED + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.BLUE + self.words_on_board[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.words_on_board[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.MAGENTA + self.words_on_board[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 60))
        print("\n")

    def _display_board(self):
        """prints the list of words in a board like fashion (5x5)"""
        print(colorama.Style.RESET_ALL)
        print(str.center("___________________________BOARD___________________________", 60))
        for i in range(len(self.words_on_board)):
            if i % 5 == 0:
                print("\n")
            print(str.center(self.words_on_board[i], 10), " ", end='')

        print(str.center("\n___________________________________________________________", 60))
        print("\n")

    def _display_key_grid(self):
        """ Print the key grid to stdout  """
        print("\n")
        print(str.center(colorama.Fore.RESET +
                         "____________________________KEY____________________________\n", 55))
        counter = 0
        for i in range(len(self.key_grid)):
            if counter >= 1 and i % 5 == 0:
                print("\n")
            if self.key_grid[i] == 'Red':
                print(str.center(colorama.Fore.RED + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Blue':
                print(str.center(colorama.Fore.BLUE + self.key_grid[i], 15), " ", end='')
                counter += 1
            elif self.key_grid[i] == 'Civilian':
                print(str.center(colorama.Fore.RESET + self.key_grid[i], 15), " ", end='')
                counter += 1
            else:
                print(str.center(colorama.Fore.MAGENTA + self.key_grid[i], 15), " ", end='')
                counter += 1
        print(str.center(colorama.Fore.RESET +
                         "\n___________________________________________________________", 55))
        print("\n")

    def get_words_on_board(self):
        """Return the list of words that represent the board state"""
        return self.words_on_board

    def get_key_grid(self):
        """Return the codemaster's key"""
        return self.key_grid
    
    def get_target_words_for_team(self, team_color: str) -> List[str]:
        """Get the target words for a specific team"""
        target_words = []
        for i, word in enumerate(self.words_on_board):
            if self.key_grid[i] == team_color and not word.startswith("*"):
                target_words.append(word)
        return target_words
    
    def capture_clue_data(self, clue: str, clue_num: int, codemaster, team_color: str):
        """Capture clue data for analysis"""
        if not self.collect_data:
            return
            
        # Get intended target words from key grid
        target_words = self.get_target_words_for_team(team_color)
        
        # Store current clue context for guess tracking
        self.current_clue = clue
        self.current_clue_number = clue_num
        self.current_codemaster = codemaster.__class__.__name__
        self.current_team_color = team_color
        self.guess_order_in_turn = 0
        
        # Create preliminary clue record (will be updated with guesses)
        clue_record = ClueRecord(
            clue_word=clue,
            clue_number=clue_num,
            codemaster_name=self.current_codemaster,
            target_words=target_words,
            guessed_words=[],  # Will be filled as guesses are made
            correct_guesses=[],  # Will be filled as guesses are made
            turn_number=self.turn_counter,
            team_color=team_color,
            game_seed=int(self.seed),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.clue_records.append(clue_record)
    
    def capture_guess_data(self, guess_word: str, guesser, card_type: str, is_correct: bool):
        """Capture individual guess data"""
        if not self.collect_data:
            return
            
        self.guess_order_in_turn += 1
        
        # Try to get confidence from guesser if available
        confidence = 0.5  # Default confidence
        if hasattr(guesser, 'last_confidence'):
            confidence = getattr(guesser, 'last_confidence', 0.5)
        elif hasattr(guesser, 'get_confidence'):
            try:
                confidence = guesser.get_confidence()
            except:
                confidence = 0.5
        
        # Create guess record
        guess_record = GuessRecord(
            guess_word=guess_word,
            guesser_name=guesser.__class__.__name__,
            clue_word=self.current_clue or "",
            clue_number=self.current_clue_number,
            is_correct=is_correct,
            card_type=card_type,
            confidence=confidence,
            turn_number=self.turn_counter,
            guess_order=self.guess_order_in_turn,
            team_color=self.current_team_color or "",
            game_seed=int(self.seed),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.guess_records.append(guess_record)
        
        # Update the current clue record with this guess
        if self.clue_records and self.current_clue:
            current_clue_record = self.clue_records[-1]
            if current_clue_record.clue_word == self.current_clue:
                current_clue_record.guessed_words.append(guess_word)
                if is_correct:
                    current_clue_record.correct_guesses.append(guess_word)
    
    def _accept_guess(self, guess_index, game_condition):
        """Enhanced function that takes in an int index called guess to compare with the key grid
        """
        guessed_word = self.words_on_board[guess_index]
        guessed_team = self.key_grid[guess_index]
        
        # Determine if guess is correct for current team
        current_team = "Red" if game_condition == GameCondition.RED_TURN else "Blue"
        is_correct = (guessed_team == current_team)
        
        # Map team to card type for data collection
        card_type_map = {
            "Red": "team" if current_team == "Red" else "opponent",
            "Blue": "team" if current_team == "Blue" else "opponent", 
            "Civilian": "civilian",
            "Assassin": "assassin"
        }
        card_type = card_type_map.get(guessed_team, "unknown")
        
        # Capture guess data
        current_guesser = self.guesser_red if current_team == "Red" else self.guesser_blue
        self.capture_guess_data(guessed_word, current_guesser, card_type, is_correct)
        
        # STRUCTURED OUTPUT FOR GUI
        print(f"GUESS_RESULT: {guessed_word}|{guessed_team}|{game_condition.name}")
        sys.stdout.flush()

        if self.key_grid[guess_index] == "Red":
            self.words_on_board[guess_index] = "*Red*"
            if self.words_on_board.count("*Red*") >= self.num_red_words:
                return GameCondition.RED_WIN
            return GameCondition.RED_TURN

        elif self.key_grid[guess_index] == "Blue":
            self.words_on_board[guess_index] = "*Blue*"
            if self.words_on_board.count("*Blue*") >= self.num_blue_words:
                return GameCondition.BLUE_WIN
            return GameCondition.BLUE_TURN

        elif self.key_grid[guess_index] == "Assassin":
            self.words_on_board[guess_index] = "*Assassin*"
            if game_condition == GameCondition.RED_TURN:
                return GameCondition.BLUE_WIN
            else:
                return GameCondition.RED_WIN

        else:
            self.words_on_board[guess_index] = "*Civilian*"
            if game_condition == GameCondition.RED_TURN:
                return GameCondition.BLUE_TURN
            else:
                return GameCondition.RED_TURN

    def write_results(self, num_of_turns):
        """Enhanced logging function with performance metrics
        writes in both the original and a more detailed new style
        """
        red_result = 0
        blue_result = 0
        civ_result = 0
        assa_result = 0

        for i in range(len(self.words_on_board)):
            if self.words_on_board[i] == "*Red*":
                red_result += 1
            elif self.words_on_board[i] == "*Blue*":
                blue_result += 1
            elif self.words_on_board[i] == "*Civilian*":
                civ_result += 1
            elif self.words_on_board[i] == "*Assassin*":
                assa_result += 1
        total = red_result + blue_result + civ_result + assa_result

        if not os.path.exists("results"):
            os.mkdir("results")

        with open("results/bot_results.txt", "a") as f:
            f.write(
                f'TOTAL:{num_of_turns} B:{blue_result} C:{civ_result} A:{assa_result} '
                f'R:{red_result} CODEMASTER_R:{self.codemaster_red.__class__.__name__} '
                f'GUESSER_R:{self.guesser_red.__class__.__name__} '
                f'CODEMASTER_B:{self.codemaster_blue.__class__.__name__} '
                f'GUESSER_B:{self.guesser_blue.__class__.__name__} '
                f'SEED:{self.seed} WINNER:{self.game_winner}\n'
            )

        # Enhanced results with performance metrics
        enhanced_results = {
            "game_name": self.game_name,
            "total_turns": num_of_turns,
            "R": red_result, "B": blue_result, "C": civ_result, "A": assa_result,
            "codemaster_red": self.codemaster_red.__class__.__name__,
            "guesser_red": self.guesser_red.__class__.__name__,
            "codemaster_blue": self.codemaster_blue.__class__.__name__,
            "guesser_blue": self.guesser_blue.__class__.__name__,
            "seed": self.seed,
            "winner": self.game_winner,
            "time_s": (self.game_end_time - self.game_start_time),
            "cmr_kwargs": {k: v if isinstance(v, (float, int, str)) else None
                          for k, v in self.cmr_kwargs.items()},
            "gr_kwargs": {k: v if isinstance(v, (float, int, str)) else None
                         for k, v in self.gr_kwargs.items()},
            "cmb_kwargs": {k: v if isinstance(v, (float, int, str)) else None
                          for k, v in self.cmb_kwargs.items()},
            "gb_kwargs": {k: v if isinstance(v, (float, int, str)) else None
                         for k, v in self.gb_kwargs.items()},
            
            # Enhanced metrics
            "red_turns_taken": self.red_turns_taken,
            "blue_turns_taken": self.blue_turns_taken,
            "total_clues_given": len(self.clue_records),
            "total_guesses_made": len(self.guess_records),
            "avg_cards_per_turn": sum(self.cards_revealed_per_turn) / len(self.cards_revealed_per_turn) if self.cards_revealed_per_turn else 0
        }

        with open("results/bot_results_new_style.txt", "a") as f:
            f.write(json.dumps(enhanced_results))
            f.write('\n')
    
    def get_collected_data(self) -> Tuple[List[ClueRecord], List[GuessRecord]]:
        """Return collected clue and guess data for analysis"""
        return self.clue_records.copy(), self.guess_records.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this game"""
        
        # Calculate clue efficiency metrics
        clue_efficiencies = []
        for clue_record in self.clue_records:
            if clue_record.revealed_cards > 0:
                efficiency = clue_record.target_cards / clue_record.revealed_cards
                clue_efficiencies.append(efficiency)
        
        # Calculate guess accuracy
        correct_guesses = sum(1 for guess in self.guess_records if guess.is_correct)
        total_guesses = len(self.guess_records)
        guess_accuracy = correct_guesses / total_guesses if total_guesses > 0 else 0
        
        # Team-specific metrics
        red_clues = [c for c in self.clue_records if c.team_color == "Red"]
        blue_clues = [c for c in self.clue_records if c.team_color == "Blue"]
        
        return {
            "avg_clue_efficiency": sum(clue_efficiencies) / len(clue_efficiencies) if clue_efficiencies else 0,
            "overall_guess_accuracy": guess_accuracy,
            "total_turns": self.turn_counter,
            "red_clues_count": len(red_clues),
            "blue_clues_count": len(blue_clues),
            "game_duration": self.game_end_time - self.game_start_time if self.game_end_time else 0,
            "words_found": {
                "red": self.words_on_board.count("*Red*"),
                "blue": self.words_on_board.count("*Blue*"),
                "civilian": self.words_on_board.count("*Civilian*"),
                "assassin": self.words_on_board.count("*Assassin*")
            }
        }

    @staticmethod
    def clear_results():
        """Delete results folder"""
        if os.path.exists("results") and os.path.isdir("results"):
            shutil.rmtree("results")

    def run(self):
        """Enhanced function that runs the codenames game between codemaster and guesser"""
        game_condition = GameCondition.RED_TURN
        turn_counter = 0

        while game_condition != GameCondition.BLUE_WIN and game_condition != GameCondition.RED_WIN:
            turn_counter += 1
            self.turn_counter = turn_counter
            cards_revealed_this_turn = 0
            turn_start_time = time.time()
            self.turn_start_times.append(turn_start_time)

            # STRUCTURED TURN OUTPUT
            if game_condition == GameCondition.RED_TURN:
                codemaster = self.codemaster_red
                guesser = self.guesser_red
                print("RED TEAM TURN")
                current_team = "Red"
                self.red_turns_taken += 1
            else:
                codemaster = self.codemaster_blue
                guesser = self.guesser_blue
                print("BLUE TEAM TURN")
                current_team = "Blue"
                self.blue_turns_taken += 1
            
            sys.stdout.flush()

            # board setup and display
            print('\n' * 2)
            words_in_play = self.get_words_on_board()
            current_key_grid = self.get_key_grid()
            codemaster.set_game_state(words_in_play, current_key_grid)
            self._display_key_grid()
            self._display_board_codemaster()

            # codemaster gives clue & number here
            clue, clue_num = codemaster.get_clue()
            
            # Capture clue data
            self.capture_clue_data(clue, clue_num, codemaster, current_team)
            
            print(f"STRUCTURED_CLUE: {codemaster.__class__.__name__}|{clue}|{clue_num}|{game_condition.name}")
            sys.stdout.flush()
            
            keep_guessing = True
            guess_num = 0
            clue_num = int(clue_num)

            print('\n' * 2)
            guesser.set_clue(clue, clue_num)

            while keep_guessing:
                guesser.set_board(words_in_play)
                
                # STRUCTURED GUESS OUTPUT
                guess_answer = guesser.get_answer()
                
                # Output the guess in structured format immediately
                if guess_answer and guess_answer != "no comparisons":
                    print(f"STRUCTURED_GUESS: {current_team}|{guess_answer}")
                    sys.stdout.flush()

                # if no comparisons were made/found than retry input from codemaster
                if guess_answer is None or guess_answer == "no comparisons":
                    break
                    
                guess_answer_index = words_in_play.index(guess_answer.upper().strip())
                game_condition_result = self._accept_guess(guess_answer_index, game_condition)
                
                cards_revealed_this_turn += 1

                # Check if the guess was correct for the current team
                if game_condition == game_condition_result:
                    # Correct guess - continue guessing
                    print('\n' * 2)
                    self._display_board_codemaster()
                    print("Keep Guessing? the clue is ", clue, clue_num)
                    keep_guessing = guesser.keep_guessing()

                    if not keep_guessing:
                        # Team chooses to stop guessing
                        if game_condition == GameCondition.RED_TURN:
                            game_condition = GameCondition.BLUE_TURN
                        elif game_condition == GameCondition.BLUE_TURN:
                            game_condition = GameCondition.RED_TURN
                else:
                    # Wrong guess or special card - turn ends
                    keep_guessing = False
                    game_condition = game_condition_result

                # If playing single team version, then it is always the red team's turn.
                if self.single_team and game_condition == GameCondition.BLUE_TURN:
                    game_condition = GameCondition.RED_TURN

                # Check for immediate game end (assassin or win condition)
                if game_condition in [GameCondition.RED_WIN, GameCondition.BLUE_WIN]:
                    break

            # Record cards revealed this turn
            self.cards_revealed_per_turn.append(cards_revealed_this_turn)

            # If game ended, break out of main loop
            if game_condition in [GameCondition.RED_WIN, GameCondition.BLUE_WIN]:
                break

        # STRUCTURED WIN OUTPUT
        if game_condition == GameCondition.RED_WIN:
            self.game_winner = "R"
            print("GAME_END: Red Team Wins!")
        else:
            self.game_winner = "B"
            print("GAME_END: Blue Team Wins!")
        
        sys.stdout.flush()

        self.game_end_time = time.time()
        self._display_board_codemaster()
        
        if self.do_log:
            self.write_results(turn_counter)
        
        print("Game Over")
        sys.stdout.flush()
        
        # Print data collection summary if enabled
        if self.collect_data and self.do_print:
            print(f"\nData Collection Summary:")
            print(f"  Clues recorded: {len(self.clue_records)}")
            print(f"  Guesses recorded: {len(self.guess_records)}")
            print(f"  Total turns: {turn_counter}")