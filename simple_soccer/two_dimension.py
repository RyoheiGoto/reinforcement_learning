import numpy as np
import csv

FIELD_WIDTH = 330
FIELD_HIGHT = 180
GOAL_LENGTH = 180

THRESHOLD = 30
FIELD_WIDTH_THRESHOLD = FIELD_WIDTH / THRESHOLD + 1
FIELD_HIGHT_THRESHOLD = FIELD_HIGHT / THRESHOLD + 1

STATE_NUM = 2
BALL_VELO_THRESHOLD = 5
STAND, FALL, WAKEUP = range(3)

TAU = 0.1
EPSILON = 0.01
ALPHA = 0.5
GAMMA = 0.5
ACTION_TIME = 5

class Soccer(object):
    def __init__(self):
        np.random.seed()
        self.Q = np.zeros([STATE_NUM, FIELD_HIGHT_THRESHOLD, FIELD_WIDTH_THRESHOLD, BALL_VELO_THRESHOLD, BALL_VELO_THRESHOLD])
        #q [action, x, y, dx, dy]
        self.do_learing = False

        self.robot_state = None #state
        self.action_count = None

        self.ball_states = None #[x, y, x_dot, y_dot]
        self.ball_x = None
        self.ball_y = None
        self.ball_dx = None
        self.ball_dy = None

        self.reward = None

    def step(self):
        ball_states = self.ball_states
        robot_state = self.robot_state

        self.decide_action(ball_states, robot_state)

        self.update_status()
        new_ball_states = [self.ball_x, self.ball_y, self.ball_dx, self.ball_dy]
        new_robot_states = self.robot_state
        reward = self.get_reward(new_ball_states, new_robot_states)

        if self.do_learing:
            self.q_learning(ball_states, new_ball_states, new_robot_states, reward)

        #return False if reward != 0 else True
        return False if self.ball_x < -10 else True

    def decide_action(self, ball_states, robot_state):
        if robot_state == FALL:
            self.do_learing = True
            if self.action_count > 0:
                self.action_count -= 1
            else:
                self.do_learing = False
                self.action_count = ACTION_TIME
                self.robot_state = WAKEUP
                return
        elif robot_state == WAKEUP:
            self.do_learing = False
            if self.action_count > 0:
                self.action_count -= 1
            else:
                self.action_count = None
                self.robot_state = STAND
                return
        else:
            self.do_learing = True
            policy = self.e_greedy(ball_states)
            prob = 0.0
            for action, policy in zip(xrange(STATE_NUM), policy):
                prob += policy
                if np.random.random() < prob:
                    self.robot_state = action
                    if action == FALL:
                        self.action_count = ACTION_TIME
                    return
            else:
                self.robot_state = STATE_NUM - 1
                self.action_count = ACTION_TIME
                return

    def e_greedy(self, ball_states):
        policy = []
        x, y, dx, dy = self.threshold(ball_states)
        quantity = [self.Q[action, x, y, dx, dy] for action in xrange(STATE_NUM)]

        for action in quantity:
            if action == max(quantity):
                policy.append(1.0 - EPSILON + EPSILON / STATE_NUM)
            else:
                policy.append(EPSILON / STATE_NUM)

        if sum(policy) == 1.0:
            return policy
        else:
            return map(lambda n: 1.0 / STATE_NUM, policy)

    def update_status(self):
        self.ball_x += self.ball_dx * TAU
        self.ball_y += self.ball_dy * TAU

    def get_reward(self, ball_staes, robot_state):
        x, y, dx, dy = ball_staes

        if robot_state == FALL and self.action_count == 1 and x > 10:
            reward = -1.0
        elif x < 0 and np.fabs(y) < GOAL_LENGTH / 2:
            if robot_state == FALL:
                reward = 1.0
            elif robot_state == STAND:
                reward = -1.0
            else:
                reward = -1.0
        elif x < 0 and np.fabs(y) > GOAL_LENGTH / 2:
            reward = 1.0
        else:
            reward = 0.0
        self.reward = reward

        return reward

    def q_learning(self, ball_states, new_ball_states, new_robot_state, reward):
        x, y, dx, dy = self.threshold(new_ball_states)
        newQ = max([self.Q[action, x, y, dx, dy] for action in xrange(STATE_NUM)])

        x, y, dx, dy = self.threshold(ball_states)
        oldQ = self.Q[new_robot_state, x, y, dx, dy]

        self.Q[new_robot_state, x, y, dx, dy] += ALPHA * (reward - oldQ + GAMMA * newQ)

    def threshold(self, state): #ball state threshold
        x, y, x_dot, y_dot = state

        #threshold x
        field_x = [i * 30 for i in xrange(FIELD_HIGHT_THRESHOLD)]
        for fx, s in zip(field_x, xrange(FIELD_HIGHT_THRESHOLD)):
            if x < fx:
                s1 = s
                break
        else:
            s1 = len(field_x) - 1

        #threshold y
        field_y = [i * 30 - FIELD_WIDTH / 2 for i in xrange(FIELD_WIDTH_THRESHOLD)]
        for fy, s in zip(field_y, xrange(FIELD_WIDTH_THRESHOLD)):
            if y < fy:
                s2 = s
                break
        else:
            s2 = len(field_y) - 1

        #threshold x_dot
        field_x_dot = [0.2 * i for i in xrange(BALL_VELO_THRESHOLD)]
        for fxd, s in zip(field_x_dot, xrange(BALL_VELO_THRESHOLD)):
            if x_dot < fxd:
                s3 = s
                break
        else:
            s3 = len(field_x_dot) - 1

        #threshold y_dot
        field_y_dot = [0.2 * i for i in xrange(BALL_VELO_THRESHOLD)]
        for fyd, s in zip(field_y_dot, xrange(BALL_VELO_THRESHOLD)):
            if y_dot < fyd:
                s4 = s
                break
        else:
            s4 = len(field_y_dot) - 1

        return s1, s2, s3, s4

    def process(self, max_episode=1000):
        f = open("ballvelo.csv", "w")
        csvwriter = csv.writer(f)
        clear = 0.0

        for episode in np.arange(0, max_episode):
            self.ball_x = np.random.randint(100, 180)
            self.ball_y = np.random.randint(-100, 100)
            self.ball_dx = -np.random.random() * 100
            self.ball_dy = np.random.choice((-np.random.random(), np.random.random())) * 100
            self.ball_states = [self.ball_x, self.ball_y, self.ball_dx, self.ball_dy]
            self.robot_state = STAND

            for step in np.arange(0, 1000000):
                if episode == 900:
                    csvwriter.writerow([step, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy, self.robot_state])
                if not self.step():
                    if self.reward > 0:
                        clear += 1.0
                    break

        print "episode: %lf\nclear: %lf(%lf%%)" % (max_episode, clear, (clear / max_episode) * 100)

        f.close()

if __name__ == '__main__':
    soccer = Soccer()
    soccer.process()

