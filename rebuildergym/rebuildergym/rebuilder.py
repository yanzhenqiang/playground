# https://pet.timetocode.org/

import copy
import math
import sys
import json
import operator
import time

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

# https://github.com/r1chardj0n3s/euclid/blob/master/euclid.py
class Vector2:
    def __init__(self, x = 0, y= 0):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Vector2(%.2f, %.2f)' % (self.x, self.y)

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    __radd__ = __add__

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        assert type(other) in (int, float)
        return Vector2(self.x * other, self.y * other)
    
    __rmul__ = __mul__

    def __imul__(self, other):
        assert type(other) in (int, float)
        self.x *= other
        self.y *= other
        return self

    def __div__(self, other):
        assert type(other) in (int, float)
        return Vector2(operator.div(self.x, other),
                       operator.div(self.y, other))

    def __truediv__(self, other):
        assert type(other) in (int, float)
        return Vector2(operator.truediv(self.x, other),
                       operator.truediv(self.y, other))

    def __neg__(self):
        return Vector2(-self.x, -self.y)

    def __abs__(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    magnitude = __abs__

    def normalize(self):
        d = self.magnitude()
        if d:
            self.x /= d
            self.y /= d
        return self

    def dot(self, other):
        assert isinstance(other, Vector2)
        return self.x * other.x + self.y * other.y

    def cross(self):
        return Vector2(self.y, -self.x)

    def rotated(self, angle_degrees):
        angle_radians = math.radians(angle_degrees)
        cos = math.cos(angle_radians)
        sin = math.sin(angle_radians)
        x = self.x * cos - self.y * sin
        y = self.x * sin + self.y * cos
        return Vector2(x, y)

    # TODO(Zhenqiang):remove this func.
    def get_angle(self):
        if (self.x == 0 and self.y == 0):
            return 0
        return math.degrees(math.atan2(self.y, self.x))

    def angle(self, other):
        """Return the angle to the vector other"""
        return math.acos(self.dot(other) / (self.magnitude()*other.magnitude()))

    def project(self, other):
        """Return one vector projected on the vector other"""
        n = other.normalized()
        return self.dot(n)*n

class MutalbeBox:
    def __init__(self, name, fix, pos_2d_m , shape_2d_m, direction_2d_m):
        self.name = name
        self.fix = fix
        self.pos_2d_m = Vector2(pos_2d_m[0], pos_2d_m[1])
        self.shape_2d_m = Vector2(shape_2d_m[0], shape_2d_m[1])
        self.direction_2d_m = Vector2(direction_2d_m[0], direction_2d_m[1])
        

    def rotate(self, angle_deg):
        if(self.fix == False):
            self.direction_2d_m = self.direction_2d_m.rotated(angle_deg)

    def translate(self, delta_vector):
        if(self.fix == False):
            self.pos_2d_m += delta_vector

    def radius(self):
        return self.shape_2d_m.magnitude()/2.0*1.1

import pygame
# TODO:change to nametuple: action_numb, shortcut, func_name, desc etc.
name2color = {
    'car': pygame.color.THECOLORS["grey"],
    'pedestrian': pygame.color.THECOLORS["blue"],
}
action2shortcut = {
    0: pygame.K_TAB,    # Choose the next object
    1: pygame.K_c,
    2: pygame.K_a,
    3: pygame.K_s,
    4: pygame.K_d,
    5: pygame.K_w,
    6: pygame.K_q,
    7: pygame.K_x,
    8: pygame.K_y,
    9: pygame.K_m,
    10: pygame.K_n,
}
shortcut2action= dict(zip(action2shortcut.values(), action2shortcut.keys()))

class RebuilderEnv(gym.Env):

  # TODO:No use
  from_pixels = False
  atari_mode = False

  width_px = 640
  height_px = 640

  def __init__(self):
    self.action_space = spaces.Discrete(16)
    if self.from_pixels:
      self.observation_space = spaces.Box(low=0, high=255,
      shape=(self.width_px, self.height_px, 3), dtype=np.uint8)
    else:
      # TODO: max number of pucks * state about 12
      high = np.array([np.finfo(np.float32).max] * 12)
      self.observation_space = spaces.Box(-high, high)
    
    # TODO:Move this to world state.
    self.view_offset_px = Vector2(0,0)
    self.view_center_px = Vector2(0,0)
    self.view_zoom = 1
    self.view_zoom_rate = 0.01
    self.length_x_m = 10
    self.px_to_m = self.length_x_m/float(self.width_px)
    self.m_to_px = (float(self.width_px)/self.length_x_m)

    # TODO:move the pygame to separate of the class.
    pygame.init()
    pygame.display.set_caption("RebuilderEnv")
    self.surface = pygame.display.set_mode((self.width_px, self.height_px))
    # ----------->  x/width_m
    # |
    # |
    # v
    # y/height_m
    self.ur_2d_m = self.screen2word(Vector2(self.width_px, 0))        
    self.surface.fill(pygame.color.THECOLORS["black"])
    pygame.display.update()

    self.pucks = []
    self.goal_pucks = []
    self.select_puck_id = 0
    self.walls = {"L_m":0.0, "R_m":self.ur_2d_m.x, "B_m":0.0, "T_m":self.ur_2d_m.y}
    self.creat_example_pucks()
    self.from_json()

  def creat_example_pucks(self):
    # Make the car
    car = MutalbeBox(name = 'car',
                     fix = True,
                     pos_2d_m = (6.0, 0.5),
                     shape_2d_m = (0.2, 0.5), 
                     direction_2d_m = (0.0, 1.0))
    self.pucks.append(car)

    # Make the pedestrian
    pedestrian = MutalbeBox(name = 'pedestrian',
                            fix = True,
                            pos_2d_m = (3.0, 0.5),
                            shape_2d_m = (0.2, 0.5), 
                            direction_2d_m = (0.0, 1.0))
    self.pucks.append(pedestrian)

  def copy_current_puck(self):
    new_puck = copy.deepcopy(self.pucks[self.select_puck_id])
    new_puck.fix = False
    new_puck.pos_2d_m += Vector2(1.0, 1.0)
    self.pucks.append(new_puck)

  def draw_pucks(self, pucks, surface, width):
    for puck in pucks:
      # Draw the example text
      if(puck.fix == True):
          font = pygame.font.SysFont("Arial", 14)
          txt_surface = font.render(puck.name, True, pygame.color.THECOLORS["white"])
          x, y = self.world2screen(puck.pos_2d_m + Vector2(-1.5, 0.1))
          surface.blit(txt_surface, [x, y]) 
      tube_vertices_2d_m = [Vector2(-0.5 * puck.shape_2d_m.x, -0.5 * puck.shape_2d_m.y),
                            Vector2(-0.5 * puck.shape_2d_m.x, 0.5 * puck.shape_2d_m.y),
                            Vector2( 0.5 * puck.shape_2d_m.x, 0.5 * puck.shape_2d_m.y),
                            Vector2( 0.5 * puck.shape_2d_m.x, -0.5 * puck.shape_2d_m.y)]
      rotated_vertices_2d_m = []
      for vertex_2d_m in tube_vertices_2d_m:
          rotated_vertices_2d_m.append(vertex_2d_m.rotated(puck.direction_2d_m.get_angle()))
      pygame.draw.polygon(surface, name2color[puck.name], self.rect_world2screen(rotated_vertices_2d_m, puck.pos_2d_m), width)

  def draw_walls(self, surface):
    # TODO(Zhenqiang):Make the wall to rect not line
    topLeft_2d_px = self.world2screen(Vector2(self.walls['L_m'], self.walls['T_m']))
    topRight_2d_px = self.world2screen(Vector2(self.walls['R_m']-0.01, self.walls['T_m']))
    botLeft_2d_px = self.world2screen(Vector2(self.walls['L_m'], self.walls['B_m']+0.01))
    botRight_2d_px = self.world2screen(Vector2(self.walls['R_m']-0.01, self.walls['B_m']+0.01))
    botLeft_2d_px_2 = self.world2screen(Vector2( self.walls['L_m'], self.walls['B_m']+1))
    botRight_2d_px_2 = self.world2screen(Vector2( self.walls['R_m']-0.01, self.walls['B_m']+1))
    pygame.draw.line(surface, pygame.color.THECOLORS["orangered1"], topLeft_2d_px,  topRight_2d_px, 1)
    pygame.draw.line(surface, pygame.color.THECOLORS["orangered1"], topRight_2d_px, botRight_2d_px, 1)
    pygame.draw.line(surface, pygame.color.THECOLORS["orangered1"], botRight_2d_px, botLeft_2d_px,  1)
    pygame.draw.line(surface, pygame.color.THECOLORS["orangered1"], botLeft_2d_px,  topLeft_2d_px,  1)
    pygame.draw.line(surface, pygame.color.THECOLORS["orangered1"], botRight_2d_px_2 , botLeft_2d_px_2,  1)

  def draw(self):
    # TODO: what 0,0,0 mean?
    self.surface.fill((0,0,0))

    self.draw_walls(self.surface)
    self.draw_pucks(self.pucks, self.surface, 3)
    self.draw_pucks(self.goal_pucks, self.surface, 1)
    # Draw a circle around select puck.
    # TODO:Move the select or not to puck.
    puck = self.pucks[self.select_puck_id]
    pygame.draw.circle(self.surface, pygame.color.THECOLORS["grey"], self.world2screen(puck.pos_2d_m), self.px_from_m(puck.radius()), width=1)
    
    pygame.display.flip()
  
  def rect_world2screen(self, vertices_2d_m, base_point_2d_m):
    vertices_2d_px = []
    for vertex_2d_m in vertices_2d_m:
        vertices_2d_px.append(self.world2screen(vertex_2d_m + base_point_2d_m))
    return vertices_2d_px

  def px_from_m(self, dx_m):
    return dx_m * self.m_to_px * self.view_zoom

  def m_from_px(self, dx_px):
    return float(dx_px) * self.px_to_m / self.view_zoom

  def screen2word(self, point_2d_px):
    x_m = (point_2d_px.x + self.view_offset_px.x) / (self.m_to_px * self.view_zoom)
    y_m = (self.height_px - point_2d_px.y + self.view_offset_px.y) / (self.m_to_px * self.view_zoom)
    return Vector2(x_m, y_m)

  def world2screen(self, point_2d_m):
    x_px = (point_2d_m.x * self.m_to_px * self.view_zoom) - self.view_offset_px.x
    y_px = (point_2d_m.y * self.m_to_px * self.view_zoom) - self.view_offset_px.y
    y_px = self.height_px - y_px
    return (int(x_px), int(y_px))

  def action_switch(self, action):
    if (action==0):
      self.select_puck_id = (self.select_puck_id + 1)%len(self.pucks)
    elif (action==1):
      self.copy_current_puck()
    elif (action==2):
      self.pucks[self.select_puck_id].translate(Vector2(-1.0, 0.0)*0.01)
    elif (action==3):
      self.pucks[self.select_puck_id].translate(Vector2(0.0, -1.0)*0.01)
    elif (action==4):
      self.pucks[self.select_puck_id].translate(Vector2(+1.0, 0.0)*0.01)
    elif (action==4):
      self.pucks[self.select_puck_id].translate(Vector2(0.0, +1.0)*0.01)
    elif (action==6):
      self.pucks[self.select_puck_id].rotate(+1.0)
    elif (action==7):
      self.pucks[self.select_puck_id].shape_2d_m += Vector2(0.01, 0.0)
    elif (action==8):
      self.pucks[self.select_puck_id].shape_2d_m += Vector2(0.0, -0.01)
    elif (action==9):
      self.pucks[self.select_puck_id].shape_2d_m += Vector2(-0.01, 0.0)
    elif (action==10):
      self.pucks[self.select_puck_id].shape_2d_m += Vector2(0.0, 0.01)

  def get_input(self):
    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            sys.exit()
        elif (event.type == pygame.KEYDOWN):
          if (event.key==pygame.K_z):
            self.view_zoom += self.view_zoom_rate * self.view_zoom
          elif (event.key==pygame.K_o):
            self.view_zoom -= self.view_zoom_rate * self.view_zoom
          elif (event.key==pygame.K_t):
            self.to_json()
          elif (event.key==pygame.K_f):
            self.from_json()
          else:
            if event.key in action2shortcut.values():
              self.action_switch(shortcut2action[event.key])

  def state(self):
    self.state = {}
    for i, puck in enumerate(self.pucks):
      puck_state = {}
      puck_state['name'] = puck.name
      puck_state['pos_2d_m'] = (puck.pos_2d_m.x, puck.pos_2d_m.y)
      puck_state['shape_2d_m'] = (puck.shape_2d_m.x, puck.shape_2d_m.y)
      puck_state['direction_2d_m'] = (puck.direction_2d_m.x, puck.direction_2d_m.y)
      puck_state['fix'] = puck.fix
      self.state[i] = puck_state
    return self.state

  def to_json(self):
    with open('test.json', 'w')as f:
      f.write(json.dumps(self.state(), indent=4))
    
  def from_json(self):
    with open('goal.json', 'r')as f:
      self.goal_state = json.loads(f.read())
    for puck_state in self.goal_state.values():
      if puck_state['fix'] == False:
        self.goal_pucks.append(MutalbeBox(**puck_state))

  def clear(self):
    self.surface.fill(pygame.color.THECOLORS["black"])
    pygame.display.update()
      
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.ale = self.game.agent_right # for compatibility for some models that need the self.ale.lives() function
    return [seed]
  
  def get_observation(self):
    if self.from_pixels:
      obs = self.render(mode='state')
    else:
      # TODO(Zhenqiang):Rearrange the state.
      obs = self.pucks
    return obs

  def discrete2box(self, n):
    # convert discrete action n into the actual triplet action
    if isinstance(n, (list, tuple, np.ndarray)): # original input for some reason, just leave it:
      if len(n) == 3:
        return n
    assert (int(n) == n) and (n >= 0) and (n < 6)
    return self.action_table[n]

  def compute_reward(self):
    # [car, center_x, center_y, shape_x, shape_y, theta] in Goal Boxes
    #   |
    #   ------------> [car, center_x, center_y, shape_x, shape_y, theta]  closest object in Current Boxes
    # [car, center_x, center_y, shape_x, shape_y, theta]
    #   |
    #   ------------> [car, center_x, center_y, shape_x, shape_y, theta]  closest object in Current Boxes
    # [car, center_x, center_y, shape_x, shape_y, theta]
    #   |
    #   ------------> [oo]  closest object in Current Boxes
    def distance_bwteen_pucks(puck1, puck2 = None):
      max_delta_pos = self.length_x_m
      delta_pos_normal = 1
      delta_angle_normal = 1
      # TODO: How to compute the distance of shape
      delta_shape_normal = 1
      if(puck2 is not None):
        # x y distance
        delta_pos_m = (puck1.pos_2d_m - puck2.pos_2d_m).magnitude()
        delta_pos_normal = delta_pos_m/max_delta_pos
        # angle distance
        delta_angle_deg = puck1.direction_2d_m.get_angle() - puck2.direction_2d_m.get_angle()
        delta_angle_normal = abs(delta_angle_deg)/360
        # shape distance
        delta_shape_m = (puck1.shape_2d_m - puck2.shape_2d_m).magnitude()
        delta_shape_normal = delta_shape_m / puck1.shape_2d_m.magnitude()
      return delta_pos_normal + delta_angle_normal + delta_shape_normal
    matched_mini_id = []
    matched_mini_distance = []
    max_distace = 3
    for goal_puck in self.goal_pucks:
      mini_id = 0
      mini_distance = max_distace
      for i, puck in enumerate(self.pucks):
        if(puck.fix): continue
        if i not in matched_mini_id:
          i_distance = distance_bwteen_pucks(puck, goal_puck)
          if(i_distance < mini_distance):
            mini_distance =  i_distance
            mini_id = i
      matched_mini_id.append(mini_id)
      matched_mini_distance.append(mini_distance)
    for i, puck in enumerate(self.pucks):
      if(puck.fix): continue
      if i not in matched_mini_id:
        matched_mini_id.append(i)
        matched_mini_distance.append(max_distace * 2)
    reward = [max_distace - item for item in matched_mini_distance]
    print(reward)
    return sum(reward)

  # TODO:misc
  def set_goal(self, goal_config):
    self.goal_state = goal_state

  def step(self, action=None, other_action=True):
    done = False
    if other_action is not None:
      self.get_input()
    else:
      self.action_switch(action)
    reward = self.compute_reward()

    # TODO(Zhenqiang):??
    if self.atari_mode:
      action = self.discrete2box(action)
      other_action = self.discrete2box(other_action)

    obs = self.get_observation()

    # if self.t >= self.t_limit:
    #   done = True

    info = {
    #   'ale.lives': self.game.agent_right.lives(),
    #   'ale.otherLives': self.game.agent_left.lives(),
    #   'otherObs': otherObs,
    #   'state': self.game.agent_right.get_observation(),
    #   'otherState': self.game.agent_left.get_observation(),
    }
    return obs, reward, done, info

  def reset(self):
    # TODO(Zheniang):reset the world state.
    return self.get_observation()

  # TODO: move the pygame inside the render
  def render(self, mode='human'):
    pass

register(
    id='Rebuilder-v0',
    entry_point='rebuildergym.rebuilder:RebuilderEnv'
)

if __name__ == '__main__':
    world2d = RebuilderEnv()
    while True:
        world2d.step()
        world2d.draw()
        # Sleep 0.1 s
        time.sleep(0.1)