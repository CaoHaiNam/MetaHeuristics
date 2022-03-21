import pygame
import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
import math
from torchvision import transforms as T
from collections import deque, namedtuple
import random
from torch.distributions import Normal

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)



WIN_SIZE = 800
ENV_SIZE = 500

VIEW = 5

MAX_SPEED = 3
MAX_SIZE = 3
FOOD_SIZE = 1

MAX_ENERGY = MAX_SIZE * 3 * 100

BLACK = torch.tensor([0,0,0]).float()
WHITE = torch.tensor([1,1,1]).float()
RED = torch.tensor([1,0,0]).float()
GREEN = torch.tensor([0,1,0]).float()
PHEROMONE = torch.tensor([0.1,0.1,0.1]).float()

GAMMA_C = 9
GAMMA_M = 10

MUTATION_RATE = 0.1

NUM_PREYS = 10
NUM_HUNTERS = 10
NUM_FOODS = 10

NUM_BARRIERS = 10


# ENV = torch.ones((ENV_SIZE+2*VIEW, ENV_SIZE+2*VIEW, 3))
# FOODS_POS = torch.zeros(NUM_FOODS, 2)

DIRECTION = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])

def draw(pos, size, color):
    ENV[pos[0]-size[0]: pos[0]+size[0], pos[1]-size[1]: pos[1]+size[1]] = color

def check_move(pos, step, size, color):
    temp = pos + step
    draw(pos, size, BLACK)

    if (ENV[temp[0]-size[0]: temp[0]+size[0], temp[1]-size[1]: temp[1]+size[1], 2] >= color[2]).any():
        draw(pos, size, color)
        return False
    else:
        draw(pos, size, color)
        return True

def move(pos, step, size, color):
    temp = pos + step
    draw(pos, size, BLACK)
    if (ENV[temp[0]-size[0]: temp[0]+size[0], temp[1]-size[1]: temp[1]+size[1], 2] >= color[2]).any():
        draw(pos, size, color)
        return False
    else:
        draw(temp, size, color)
        return True

def draw_win(display):
    surf = pygame.surfarray.make_surface((ENV*255).cpu().numpy())
    # surf = pygame.surfarray.make_surface((ENV[VIEW: ENV_SIZE+VIEW, VIEW: ENV_SIZE+VIEW]*255).cpu().numpy())
    surf = pygame.transform.scale(surf, (WIN_SIZE, WIN_SIZE))
    display.blit(surf, (0, 0))
    pygame.display.update()

class Agent(object):
    """docstring for Fish"""
    def __init__(self, speed=None, size=None, p=None):
        super(Agent, self).__init__()
        if speed is not None:
            self.speed = speed
        else:
            self.speed = np.random.randint(1, MAX_SPEED)

        if size is not None:
            self.size = size
        else:
            self.size = np.random.randint(2, MAX_SIZE)

        if p is not None:
            self.p = p
        else:
            self.p = np.random.rand(2)
            self.p = self.p / sum(self.p)

        self.re_spawn()


    def re_spawn(self):
        self.pos = torch.randint(VIEW, ENV_SIZE+VIEW, (2,))
        self.energy = MAX_ENERGY
        self.total_energy = 0
        self.color = torch.tensor([0, self.speed/MAX_SPEED, self.size/MAX_SIZE]).float()
        draw(self.pos, (self.size, self.size), self.color)


    def move(self):
        if self.energy <= 0:
            return

        # self.energy -= self.speed + self.size * 2

        # m = np.random.choice(['heuristic', 'random'], p=self.p)
        m = 'heuristic'
        if m == 'heuristic':
            rewards = self.compute_reward(self.pos.expand(4, 2) + DIRECTION*self.speed)
            for i in range(4):
                if not check_move(self.pos, DIRECTION[i]*self.speed, (self.size, self.size), self.color):
                    rewards[i] = -99999
            _, action = rewards.max(0)
        elif m == 'random':
            action = np.random.randint(0, 4)

        # draw(self.pos, self.size, BLACK)
        if move(self.pos, DIRECTION[action]*self.speed, (self.size, self.size), self.color):
            self.pos += DIRECTION[action]*self.speed
        # self.pos = torch.clamp(self.pos, VIEW+self.size, ENV_SIZE+VIEW-self.size)

        # draw(self.pos, self.size, self.color)

        return

    def eat(self, food):
        if self.energy <= 0:
            return False

        if (torch.abs(self.pos-food.pos) <= self.size+food.size).all():
            return True

        return False


class Hunter(Agent):
    """docstring for Hunter"""
    def __init__(self, speed=None, size=None, p=None):
        super(Hunter, self).__init__(speed, size, p)
        # self.color = RED

    def compute_reward(self, pos):
        distance = (pos.unsqueeze(1).expand(4, PREYS_POS.shape[0], 2) - PREYS_POS.unsqueeze(0).expand(4, PREYS_POS.shape[0], 2)).abs().sum(2)
        # print(torch.exp(-distance/10).sum(1))
        return torch.exp(-distance/10).sum(1)

class Prey(Agent):
    """docstring for Hunter"""
    def __init__(self, speed=None, size=None, p=None):
        super(Prey, self).__init__(speed, size, p)
        # self.color = GREEN

    def compute_reward(self, pos):
        distance_hunters = (pos.unsqueeze(1).expand(4, HUNTERS_POS.shape[0], 2) - HUNTERS_POS.unsqueeze(0).expand(4, HUNTERS_POS.shape[0], 2)).abs().sum(2)
        distance_foods = (pos.unsqueeze(1).expand(4, FOODS_POS.shape[0], 2) - FOODS_POS.unsqueeze(0).expand(4, FOODS_POS.shape[0], 2)).abs().sum(2)
        return torch.exp(-distance_foods/10).sum(1)-torch.exp(-distance_hunters/10).sum(1)
       

def eat(fish1, fish2):

    if (torch.abs(fish1.pos-fish2.pos) <= fish1.size+fish2.size).all():
        if fish1.size < fish2.size:
            draw(fish1.pos, (fish1.size, fish1.size), BLACK)
            return 1
        elif fish1.size > fish2.size:
            draw(fish2.pos, (fish2.size, fish2.size), BLACK)
            return 2

    return 0


def duplicate(parent):
    offspring = Fish()
    offspring.pos = parent.pos.clone()

    if np.random.rand() > MUTATION_RATE:
        offspring.speed = parent.speed

    if np.random.rand() > MUTATION_RATE:
        offspring.size = parent.size

    # for pc, pp in zip(offspring.net.parameters(), parent.net.parameters()):
    # pp = parent.p
    # pc = offspring.p

    # u = np.random.rand(pc.shape[0])
    # beta1 = (2*u) ** (1/(GAMMA_M+1)) - 1
    # beta2 = 1 - (1/(2-2*u)) ** (1/(GAMMA_M+1))
    # pc[u<=0.5] = pp[u<=0.5] + beta1[u<=0.5] * pp[u<=0.5]
    # pc[u>0.5] = pp[u>0.5] + beta2[u>0.5] * (1-pp[u>0.5])


    # offspring.p = offspring.p / sum(offspring.p)
    return offspring


def game():
    pygame.init()
    running = True
    clock = pygame.time.Clock()
    display = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
    # pygame.mouse.set_visible(False)

    global HUNTERS_POS, PREYS_POS, FOODS_POS, ENV, BARRIERS
    ENV = torch.ones((ENV_SIZE+2*VIEW, ENV_SIZE+2*VIEW, 3))
    ENV[VIEW: ENV_SIZE+VIEW, VIEW: ENV_SIZE+VIEW] = BLACK

    BARRIERS_SIZE = np.array([[1, np.random.randint(1, 100)] for i in range(NUM_BARRIERS)] 
                            + [[np.random.randint(1, 100), 1] for i in range(NUM_BARRIERS)])


    BARRIERS_POS = np.array([[np.random.randint(BARRIERS_SIZE[i][0], ENV_SIZE-BARRIERS_SIZE[i][0]),
                            np.random.randint(BARRIERS_SIZE[i][1], ENV_SIZE-BARRIERS_SIZE[i][1])] 
                                for i in range(NUM_BARRIERS*2)])

    for i in range(NUM_BARRIERS*2):
        draw(BARRIERS_POS[i], BARRIERS_SIZE[i], WHITE)

    hunters = [Hunter(speed=2, size=3) for _ in range(NUM_HUNTERS)]
    preys = [Prey(speed=3, size=2) for _ in range(NUM_PREYS)]
    foods = [Agent(speed=0, size=1) for _ in range(NUM_FOODS)]

    HUNTERS_POS = torch.stack([hunter.pos for hunter in hunters])
    PREYS_POS = torch.stack([prey.pos for prey in preys])
    FOODS_POS = torch.stack([food.pos for food in foods])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        clock.tick(10)
        i = 0
        for food in foods:
            draw(food.pos, (food.size, food.size), food.color)
        while True:
            if i >= len(preys):
                break

            preys[i].move()
            j = 0
            while True:
                if j >= len(foods):
                    break

                if preys[i].eat(foods[j]):
                    draw(foods[j].pos, (foods[j].size, foods[j].size), BLACK)
                    foods.pop(j)
                    j -= 1
                    foods.append(Agent(speed=0, size=1))
                    FOODS_POS = torch.stack([food.pos for food in foods])
                
                j += 1

            i += 1

        PREYS_POS = torch.stack([prey.pos for prey in preys])

        i = 0
        while True:
            if i >= len(hunters):
                break

            hunters[i].move()
            j = 0
            while True:
                if j >= len(preys):
                    break

                if hunters[i].eat(preys[j]):
                    draw(preys[j].pos, (preys[j].size, preys[j].size), BLACK)
                    preys.pop(j)
                    j -= 1
                    preys.append(Prey(speed=3, size=2))
                    PREY_POS = torch.stack([prey.pos for prey in preys])
                
                j += 1

            # if fishes[i].energy <= 0:
            #     draw(fishes[i].pos, (fishes[i].size, fishes[i].size), BLACK)
            #     fishes.pop(i)
            #     i -= 1
            
            i += 1

        HUNTERS_POS = torch.stack([hunter.pos for hunter in hunters])

            
        draw_win(display)
        


    pygame.quit()

if __name__ == '__main__':
    game()