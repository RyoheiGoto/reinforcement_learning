import numpy as np
import matplotlib.pyplot as plt
import csv

FIELD_WIDTH = 396
FIELD_HIGHT = 180
GOAL_LENGTH = 180

THRESHOLD = 36
FIELD_WIDTH_THRESHOLD_NUM = FIELD_WIDTH / THRESHOLD + 1
FIELD_WIDTH_THRESHOLD = [Y * THRESHOLD - FIELD_WIDTH / 2.0 for Y in xrange(FIELD_WIDTH_THRESHOLD_NUM)]
FIELD_HIGHT_THRESHOLD_NUM = FIELD_HIGHT / THRESHOLD + 1
FIELD_HIGHT_THRESHOLD = [X * THRESHOLD for X in xrange(FIELD_HIGHT_THRESHOLD_NUM)]

BALL_VELO_X_THRESHOLD = [X * 100.0 for X in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0]]
BALL_VELO_X_THRESHOLD_NUM = len(BALL_VELO_X_THRESHOLD) + 1
BALL_VELO_Y_THRESHOLD = [Y * 100.0 / 5.0 for Y in [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
BALL_VELO_Y_THRESHOLD_NUM = len(BALL_VELO_Y_THRESHOLD) + 1

ROBOT_STATES = 3
STAND, FALL_LEFT, FALL_RIGHT = range(ROBOT_STATES)

TAU = 0.2
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.5

FALL_TIME = 3.0

class Soccer(object):
    def __init__(self):
        np.random.seed()
        self.Q = np.zeros([ROBOT_STATES, FIELD_HIGHT_THRESHOLD_NUM, FIELD_WIDTH_THRESHOLD_NUM, BALL_VELO_X_THRESHOLD_NUM, BALL_VELO_Y_THRESHOLD_NUM])
        #Q[robot state, ball x, ball y, ball dx, ball dy]

        self.robot_state = None
        self.fall_count = None

        self.ball_states = None #[x, y, dx, dy]

        self.result = None

    def step(self):
        ball_states = self.ball_states
        robot_state = self.robot_state

        self.decide_action(ball_states, robot_state)

        self.update_status()
        new_ball_states = self.ball_states
        new_robot_states = self.robot_state

        reward, result = self.get_reward(new_ball_states, new_robot_states)

        self.q_learning(ball_states, new_ball_states, new_robot_states, reward)

        if not result:
            self.result = reward
            return False
        else:
            return True

    def decide_action(self, ball_states, robot_state):
        if robot_state == (FALL_LEFT or FALL_RIGHT):
            self.fall_count -= 1
        else:
            policy = self.e_greedy(ball_states)
            prob = 0.0
            for action, policy in zip(xrange(ROBOT_STATES), policy):
                prob += policy
                if np.random.random() < prob:
                    self.robot_state = action
                    if action == (FALL_LEFT or FALL_RIGHT):
                        self.fall_count = FALL_TIME
                    break
            else:
                self.robot_state = STAND

    def e_greedy(self, ball_states):
        policy = []
        _x, _y, _dx, _dy = self.threshold(ball_states)
        quantity = [self.Q[action, _x, _y, _dx, _dy] for action in xrange(ROBOT_STATES)]

        if sum(quantity) == 0:
            return map(lambda n: 1.0 / ROBOT_STATES, quantity)
        else:
            for action in xrange(len(quantity)):
                if action == quantity.index(max(quantity)):
                    policy.append(1.0 - EPSILON + EPSILON / ROBOT_STATES)
                else:
                    policy.append(EPSILON / ROBOT_STATES)

        if sum(policy) == 1.0:
            return policy
        else:
            return map(lambda n: 1.0 / ROBOT_STATES, policy)

    def update_status(self):
        ball_x, ball_y, ball_dx, ball_dy = self.ball_states

        ball_x += ball_dx * TAU
        ball_y += ball_dy * TAU

        self.ball_states = [ball_x, ball_y, ball_dx, ball_dy]

    def get_reward(self, ball_staes, robot_state):
        _x, _y, _dx, _dy = self.threshold(ball_staes)
        reward = 0.0
        result = True

        if robot_state == STAND:
            if _x == 1 and _y == 6: #ball clear
                reward = 5.0
                result = False
            else:
                reward = 1.0
        elif robot_state == FALL_LEFT:
            if _x == 1 and (4 <= _y <= 5): #ball clear
                reward = 5.0
                result = False
            elif self.fall_count <= 0:
                reward = -5.0
                result = False
            else:
                reward = -5.0
        elif robot_state == FALL_RIGHT:
            if _x == 1 and (7 <= _y <= 8): #ball clear
                reward = 1.0
                result = False
            elif self.fall_count <= 0:
                reward = -5.0
                result = False
            else:
                reward = -5.0

        if _x == 0 and (4 <= _y <= 8): #ball in goal
            reward = -10.0
            result = False
        elif (_x == 0 or _x == 6) and (_y < 4 or _y > 8): #ball out of line
            reward = 1.0
            result = False

        return reward, result

    def q_learning(self, ball_states, new_ball_states, new_robot_state, reward):
        x, y, dx, dy = self.threshold(new_ball_states)
        newQ = max([self.Q[action, x, y, dx, dy] for action in xrange(ROBOT_STATES)])

        x, y, dx, dy = self.threshold(ball_states)
        oldQ = self.Q[new_robot_state, x, y, dx, dy]

        self.Q[new_robot_state, x, y, dx, dy] += ALPHA * (reward - oldQ + GAMMA * newQ)

    def threshold(self, state): #ball state threshold
        x, y, dx, dy = state

        #threshold x
        for field, num in zip(FIELD_HIGHT_THRESHOLD, xrange(FIELD_HIGHT_THRESHOLD_NUM)):
            if x < field:
                th_x = num
                break
        else:
            th_x = FIELD_HIGHT_THRESHOLD_NUM - 1

        #threshold y
        for field, num in zip(FIELD_WIDTH_THRESHOLD, xrange(FIELD_WIDTH_THRESHOLD_NUM)):
            if y < field:
                th_y = num
                break
        else:
            th_y = FIELD_WIDTH_THRESHOLD_NUM - 1

        #threshold x_dot
        for velo, num in zip(BALL_VELO_X_THRESHOLD, xrange(BALL_VELO_X_THRESHOLD_NUM)):
            if dx < velo:
                th_dx = num
                break
        else:
            th_dx = BALL_VELO_X_THRESHOLD_NUM - 1

        #threshold y_dot
        for velo, num in zip(BALL_VELO_Y_THRESHOLD, xrange(BALL_VELO_Y_THRESHOLD_NUM)):
            if dy < velo:
                th_dy = num
                break
        else:
            th_dy = BALL_VELO_Y_THRESHOLD_NUM - 1

        return th_x, th_y, th_dx, th_dy

    def process(self, max_episode=10000):
        f = open("ballvelo.csv", "w")
        csvwriter = csv.writer(f)
        clear = 0.0

        for episode in np.arange(0, max_episode):
            ball_x = np.random.randint(100, 170)
            ball_y = np.random.randint(-60, 60)
            ball_dx = -np.random.random() * 100
            ball_dy = np.random.choice((-np.random.random() / 5.0, np.random.random() / 5.0)) * 100
            self.ball_states = [ball_x, ball_y, ball_dx, ball_dy]
            #self.ball_states = [150, 20, -10, 0]
            self.robot_state = STAND

            for step in np.arange(0, 100000):
                if episode == 9000:
                    csvwriter.writerow([step + 1, self.ball_states[0], self.ball_states[1], self.ball_states[2], self.ball_states[3], self.robot_state])
                if not self.step():
                    if self.result > 0:
                        clear += 1
                    break

        print "episode: %lf\nclear: %lf(%lf%%)" % (max_episode, clear, (clear / max_episode) * 100)

        f.close()

class Plotcsv(Soccer):
    def __init__(self, filename):
        Soccer.__init__(self)
        self.episode = np.loadtxt(filename, delimiter=",")
        self.field = np.zeros([FIELD_HIGHT_THRESHOLD_NUM, FIELD_WIDTH_THRESHOLD_NUM])
        self.__process()

    def __process(self):
        FIELD, BALL = range(2)
        field = np.zeros_like(self.field)
        for episode in self.episode:
            time, ball_x, ball_y, ball_dx, ball_dy, robot = episode
            x, y, dx, dy = self.threshold([ball_x, ball_y, ball_dx, ball_dx])
            field[x, y] = BALL

        plt.clf()
        plt.imshow(field, interpolation="none")
        plt.show()

if __name__ == '__main__':
    soccer = Soccer()
    soccer.process()

    Plotcsv("ballvelo.csv")
