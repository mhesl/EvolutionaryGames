import copy
import math
import random

from player import Player
import numpy as np
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):
        sigma = 0.005 if self.mode == 'thrust' else 0.5
        moue = 0
        rp = 0.2 if self.mode == 'thrust' else 0.1
        if random.random() <= rp:
            child.nn.W1 += np.random.normal(loc=moue, scale=sigma, size=child.nn.W1.shape)
        if random.random() <= rp:
            child.nn.W2 += np.random.normal(loc=moue, scale=sigma, size=child.nn.W2.shape)
        if random.random() <= rp:
            child.nn.B1 += np.random.normal(loc=moue, scale=sigma, size=child.nn.B1.shape)
        if random.random() <= rp:
            child.nn.B2 += np.random.normal(loc=moue, scale=sigma, size=child.nn.B2.shape)
        if self.mode == 'thrust':
            if random.random() <= rp:
                child.nn.W3 += np.random.normal(loc=moue, scale=sigma, size=child.nn.W3.shape)
            if random.random() <= rp:
                child.nn.B3 += np.random.normal(loc=moue, scale=sigma, size=child.nn.B3.shape)

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            prev_players.sort(key=lambda x: x.fitness, reverse=True)
            new_players = []
            top_part_division = 2
            for i in range(int(num_players/top_part_division)):
                for j in range(top_part_division):
                    new_player = copy.deepcopy(prev_players[i])
                    new_player.nn.W1 = (prev_players[i].nn.W1 + prev_players[i + j + 1].nn.W1) / 2
                    new_player.nn.W2 = (prev_players[i].nn.W2 + prev_players[i + j + 1].nn.W2) / 2
                    new_player.nn.B1 = (prev_players[i].nn.B1 + prev_players[i + j + 1].nn.B1) / 2
                    new_player.nn.B2 = (prev_players[i].nn.B2 + prev_players[i + j + 1].nn.B2) / 2
                    if self.mode == 'thrust':
                        new_player.nn.W3 = (prev_players[i].nn.W3 + prev_players[i + j + 1].nn.W3) / 2
                        new_player.nn.B3 = (prev_players[i].nn.B3 + prev_players[i + j + 1].nn.B3) / 2
                    self.mutate(new_player)
                    new_players.append(new_player)

            return new_players

    def next_population_selection(self, players, num_players):
        LEN = len(players)
        players.sort(key=lambda x: x.fitness, reverse=True)
        next_generation = [copy.deepcopy(players[0])]
        for i in range(num_players - 1):
            rand_param = random.random()
            index = int(((4 * LEN * (LEN + 1) * rand_param + 1) ** 0.5 - 1) / 2)
            next_generation.append(copy.deepcopy(players[num_players - index - 1]))
        return next_generation
