import os
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import pygame
import sys

class ZombieCampEnvironment(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, width=5, height=5, num_supplies=3, num_zombies=3, num_walls=2, num_rocks=2):
        super(ZombieCampEnvironment, self).__init__()

        self.total_reward = 0
        self.width = width
        self.height = height
        self.num_supplies = num_supplies
        self.num_zombies = num_zombies
        self.num_walls = num_walls
        self.num_rocks = num_rocks

        # 0: right, 1: left, 2: up, 3: down
        self.action_space = spaces.Discrete(4)
        # 0: horizontal position, 1: vertical position, 2: current position, 3: number of supplies collected, 4: right, 5: left, 6: top, 7: bottom 
        self.observation_space = spaces.MultiDiscrete([self.width, self.height, 6, self.num_supplies + 1, 6, 6, 6, 6])

        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.agent_position = [0, 0]

        pygame.init()

        self.cell_size = int(900 / max(self.width, self.height))
        self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
        pygame.display.set_caption("Zombie Camp")

        # load images
        self.assets = {
            'agent': pygame.image.load(os.path.join('assets', 'agent.png')),
            'supply': pygame.image.load(os.path.join('assets', 'supply.png')),
            'zombie': pygame.image.load(os.path.join('assets', 'zombie.png')),
            'door': pygame.image.load(os.path.join('assets', 'door.png')),
            'wall': pygame.image.load(os.path.join('assets', 'wall.png')),
            'rock': pygame.image.load(os.path.join('assets', 'rock.png')),
        }

        # resize the images to fit the cells
        for key in self.assets:
            self.assets[key] = pygame.transform.scale(self.assets[key], (self.cell_size, self.cell_size))

        # load the grass image
        self.background = pygame.image.load(os.path.join('assets', 'grass.png'))
        self.background = pygame.transform.scale(self.background, (self.width * self.cell_size, self.height * self.cell_size))

        # 1: supplies, 2: zombies, 3: exit, 4: walls. 5: rocks
        self.supplies = []
        self.zombies = []
        self.walls = []
        self.rocks = []
        self.exit_position = ()
        self.initial_agent_position = ()

        self._create_objects()
        self._position_objects()

        self.reset()

    def _create_objects(self):
        # Place zombies
        for _ in range(self.num_zombies):
            while True:
                x, y = np.random.randint(self.width), np.random.randint(self.height)
                if (x, y) not in self.zombies:
                    self.zombies.append((x, y))
                    break

        # Place supplies
        for _ in range(self.num_supplies):
            while True:
                x, y = np.random.randint(self.width), np.random.randint(self.height)
                if (x, y) not in self.zombies and (x, y) not in self.supplies:
                    self.supplies.append((x, y))
                    break

        # Place walls
        for _ in range(self.num_walls):
            while True:
                x, y = np.random.randint(self.width), np.random.randint(self.height)
                if (x, y) not in self.zombies and (x, y) not in self.supplies and (x, y) not in self.walls:
                    self.walls.append((x, y))
                    break

        # Place rocks
        for _ in range(self.num_rocks):
            while True:
                x, y = np.random.randint(self.width), np.random.randint(self.height)
                if (x, y) not in self.zombies and (x, y) not in self.supplies and (x, y) not in self.walls and (x, y) not in self.rocks:
                    self.rocks.append((x, y))
                    break

        # Place the door
        while True:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if (x, y) not in self.zombies and (x, y) not in self.supplies and (x, y) not in self.walls and (x, y) not in self.rocks:
                self.exit_position = (x, y)
                break

        # Place the initial agent position
        while True:
            x, y = np.random.randint(self.width), np.random.randint(self.height)
            if (x, y) not in self.zombies and (x, y) not in self.supplies and (x, y) not in self.walls and (x, y) not in self.rocks and (x, y) != self.exit_position:
                self.initial_agent_position = (x, y)
                break

    def reset(self, *, seed=None, options=None):
        self.grid = np.zeros((self.height, self.width), dtype=np.int32)
        self.total_reward = 0   

        self._position_objects() 

        return self._get_observation(), {}
    
    def _position_objects(self):
        self.agent_position = list(self.initial_agent_position)

        for x, y in self.supplies:
            self.grid[y, x] = 1

        for x, y in self.zombies:
            self.grid[y, x] = 2

        self.grid[self.exit_position[1], self.exit_position[0]] = 3

        for x, y in self.walls:
            self.grid[y, x] = 4

        for x, y in self.rocks:
            self.grid[y, x] = 5

    def _get_observation(self):
        observation = np.zeros(8, dtype=np.int32)
        x, y = self.agent_position

        # width
        observation[0] = x
        # height
        observation[1] = y
        # current position
        observation[2] = self.grid[y, x]
        # supplies collected
        observation[3] = sum(1 for x, y in self.supplies if self.grid[y, x] == 0)

        # right
        if x < self.width - 1:
            observation[4] = self.grid[y, x + 1]
        else:
            observation[4] = -1

        # left
        if x > 0:
            observation[5] = self.grid[y, x - 1]
        else:
            observation[5] = -1

        # top
        if y > 0:
            observation[6] = self.grid[y - 1, x]
        else:
            observation[6] = -1

        # bottom
        if y < self.height - 1:
            observation[7] = self.grid[y + 1, x]
        else:
            observation[7] = -1

        return observation

    def step(self, action):
        x, y = self.agent_position
        if action == 0 and x < self.width - 1 and self.grid[y, x + 1] not in [4, 5]:  # right
            x += 1
        elif action == 1 and x > 0 and self.grid[y, x - 1] not in [4, 5]:  # left
            x -= 1
        elif action == 2 and y > 0 and self.grid[y - 1, x] not in [4, 5]:  # up
            y -= 1
        elif action == 3 and y < self.height - 1 and self.grid[y + 1, x] not in [4, 5]:  # down
            y += 1

        self.agent_position = [x, y]

        reward = -1
        done = False
        truncated = False
        collected_supplies = sum(1 for x, y in self.supplies if self.grid[y, x] == 0)

        # reward for the supply
        if self.grid[y, x] == 1:
            reward += 10
            self.total_reward += 10
            self.grid[y, x] = 0
        # reward if a zombie is encoutered    
        elif self.grid[y, x] == 2:
            reward += -100
            self.total_reward = 0
            done = True
        # reward for getting to the door    
        elif self.grid[y, x] == 3:
            if collected_supplies == len(self.supplies):
                reward += 100
                done = True
            else:
                reward -= 10
                done = True

        return self._get_observation(), reward, done, truncated, {}

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # draw the grass
        self.screen.blit(self.background, (0, 0))

        # draw the objects
        for i in range(self.width):
            for j in range(self.height):
                x = i * self.cell_size
                y = j * self.cell_size
                cell_value = self.grid[j, i]

                if cell_value == 1:
                    self.screen.blit(self.assets['supply'], (x, y))
                elif cell_value == 2:
                    self.screen.blit(self.assets['zombie'], (x, y))
                elif cell_value == 3:
                    self.screen.blit(self.assets['door'], (x, y))
                elif cell_value == 4:
                    self.screen.blit(self.assets['wall'], (x, y))
                elif cell_value == 5:
                    self.screen.blit(self.assets['rock'], (x, y))

                pygame.draw.rect(self.screen, (75, 75, 75), (x, y, self.cell_size, self.cell_size), 1)

        # draw the agent
        ax, ay = self.agent_position
        self.screen.blit(self.assets['agent'], (ax * self.cell_size, ay * self.cell_size))

        pygame.display.flip()

register(
    id='ZombieCampEnvironment',
    entry_point='zombie_camp:ZombieCampEnvironment'
)