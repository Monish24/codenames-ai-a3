from players.codemaster import Codemaster
from gensim.downloader import load
import numpy as np
from gensim.parsing.preprocessing import STOPWORDS
from scipy.spatial.distance import cosine
from gensim.parsing.preprocessing import stem_text


class AICodemaster(Codemaster):
    def __init__(self, team="Red", model="glove-wiki-gigaword-300",threshold = 0.3):
        super().__init__()
        self.team = team
        self.threshold = threshold

        print(f"Loading embedding model on codemaster {team} '{model}'...")

        self.model = load(model)

    #Change to lemmatization need other library
    def stem(word):
        return stem_text(word)

    def set_game_state(self, words, maps):
        self.words = words
        self.maps = maps

    def get_remaining_options(self):
        # Converts the words and map variables into a more gpt-friendly text format
        red, blue, civilian, assassin = [], [], [], []
        for i in range(len(self.words)):
            if self.words[i][0] == '*':
                continue
            if self.maps[i] == "Red":
                red.append(self.words[i])
            if self.maps[i] == "Blue":
                blue.append(self.words[i])
            if self.maps[i] == "Civilian":
                civilian.append(self.words[i])
            if self.maps[i] == "Assassin":
                assassin.append(self.words[i])

        return red, blue, civilian, assassin
    
    #Agglomerative clustering, all words start in its own cluster and start forming clusters if sim between 2 clusters is > threshold
    #stop when no new clusters are formed
    def cluster_team_words(self, team_words, threshold=0.5):
        clusters = [[word] for word in team_words if word in self.model]

        changed = True
        while changed:
            changed = False
            new_clusters = []
            used = [False] * len(clusters)

            for i in range(len(clusters)):
                if used[i]:
                    continue
                merged = clusters[i]
                used[i] = True

                for j in range(i + 1, len(clusters)):
                    if used[j]:
                        continue
                    vec_i = np.mean([self.model[w] for w in merged], axis=0)
                    vec_j = np.mean([self.model[w] for w in clusters[j]], axis=0)

                    sim = 1 - cosine(vec_i, vec_j)
                    if sim > threshold:
                        merged += clusters[j]
                        used[j] = True
                        changed = True

                new_clusters.append(merged)

            clusters = new_clusters
        return clusters
    
    def get_clue(self):
        print(f"Codemaster {self.team} getting clue")
        red, blue, civilian, assassin = self.get_remaining_options()
        assassin = assassin[0].lower()
        if self.team == "Red":
            team_words = [x.lower() for x in red]
            opponent_words = [x.lower() for x in blue]
        else:
            team_words = [x.lower() for x in blue]
            opponent_words = [x.lower() for x in red]
        civilian = [x.lower() for x in civilian]

        clusters = self.cluster_team_words(team_words)


        best_clue = None
        best_target_words = []
        best_score = -np.inf
        forbidden = set(STOPWORDS) | set(self.words)

        for cluster in clusters:
            stemmed_forbidden = set(stem_text(w) for w in (forbidden | set(cluster)))

            cluster_vec = np.mean([self.model[w] for w in cluster], axis=0)

            candidates = [
                word for word, score in self.model.similar_by_vector(cluster_vec, topn=20)
                if word not in forbidden and word.isalpha() and stem_text(word) not in stemmed_forbidden]

            for candidate in candidates:
                # Calculate similarity to team words
                similar_team_words = [w for w in team_words if w in self.model and self.model.similarity(candidate, w) >= self.threshold]
                if len(similar_team_words) < 1:
                    continue

                # Penalize clue if it's similar to bad words
                danger_score = 0
                if assassin in self.model:
                    danger_score += 5.0 * self.model.similarity(candidate, assassin)
                danger_score += sum(self.model.similarity(candidate, w) for w in opponent_words if w in self.model)
                danger_score += 0.5 * sum(self.model.similarity(candidate, w) for w in civilian if w in self.model)

                # Reward = average similarity to selected team words - danger
                sim_score = np.mean([self.model.similarity(candidate, w) for w in similar_team_words])
                final_score = sim_score * len(similar_team_words) - danger_score

                if final_score > best_score:
                    best_score = final_score
                    best_clue = candidate
                    best_target_words = similar_team_words
            if len(best_target_words) == len(team_words):  # Early exit if we hit max not going to happen but just in case
                break
        print(f"Codemaster {self.team} clue is: {best_clue} {len(best_target_words)}, this clue was given because of the similarity with: {best_target_words}")

        return best_clue, len(best_target_words)
    
