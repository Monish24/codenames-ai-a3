# Codenames AI
This project implements a framework for simulating and evaluating various AI agents playing the game of Codenames. It includes multiple codemaster and guesser strategies, a graphical user interface (GUI) for visualizing games and tournaments using the TrueSkill ranking system.

# Project Structure
In the folder players you can find all our agents and a file called cm_wordlist.txt that contains 395 words from which the board will be created.

In the folder results single games will be saved.

In the folder tournament_results tournament results will be saved.

In the main folder codenames apart from this three folders we can find, different files to run a single game, tournaments or the gui.

# How to Run a Single Game
To run a single game you can either run the python script run_game.py using the command 
```
python run_game.py
```
 or first run the gui and start a game there.

# How to Run a Tournament
To run a tournament you can either run the python script run_believability_tournament.py using the command:
```
python run_believability_tournament.py
```
or first run the gui and start a tournament there.

# How to Run the GUI
To run the GUI you must run the python script codenames_gui.py using the command:
```
python codenames_gui.py
```
from there you can select agent to play single games or run a tournament.