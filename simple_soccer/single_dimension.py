import numpy as np

GOAL, FIELD, BALL, ROBOT = range(4)
STAND, FALL, WAKEUP = range(3)
FAILED, SAVED, CONTINUE = range(3)

ball_velo = 1
epsilon = 0.1
gamma = 0.9

class Soccer(object):
    def __init__(self):
        np.random.seed()
        self.field_default = [BALL, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, FIELD, ROBOT, GOAL]
        self.field_status = None
        self.robot_state = None
        self.robot_state_old = None
        self.Q = np.zeros([3, len(self.field_default)])
        self.D = []
        self.num_saved = 0
        self.robot_list = None

    def game(self):
        self.field_status = self.field_default[:]
        self.robot_state = STAND
        self.robot_list = []
        d = []

        for time in range(11):
            self.robot_list.append(self.robot_state)
            #self.plotresult(self.field_status)
            self.robot_state_old = self.robot_state
            self.robot_state = self.robot_update()
            status = self.move_ball()
            reward = self.get_reward()
            d.append([time, self.robot_state, reward])
            if time >= 10:
                if status == FAILED:
                    pass
                if status == SAVED:
                    self.num_saved += 1
        #self.plotresult(self.field_status)
        self.D.append(d)

    def get_reward(self):
        if self.field_status.index(BALL) != 11:
            if self.robot_state == (STAND or WAKEUP) and self.field_status.index(BALL) != 10:
                reward = -1.0
            else:
                reward = 0.0
        else:
            reward = -1.0

        return reward

    def robot_update(self):
        if self.robot_state_old == STAND:
            q = [self.Q[action, self.field_status.index(BALL)] for action in xrange(2)]
            policy = map(lambda n: 0.0, xrange(2))
            for action in xrange(2):
                if q[action] == max(q):
                    policy[action] = 1.0 - epsilon + epsilon / 2.0
                else:
                    policy[action] = epsilon / 2.0
            if sum(policy) != 1.0:
                policy = map(lambda n: 1.0 / 2.0, policy)

            prob = 0.0
            for action in xrange(2):
                prob += policy[action]
                if np.random.random() < prob:
                    return action
        if self.robot_state_old == FALL:
            old = self.robot_list[-2]
            if old == FALL:
                return WAKEUP
            else:
                return FALL
        if self.robot_state_old == WAKEUP:
            old = self.robot_list[-2]
            if old == WAKEUP:
                return STAND
            else:
                return WAKEUP

    def move_ball(self):
        pos = self.field_status.index(BALL)
        if self.field_status[pos + ball_velo] == ROBOT:
            if self.result() == SAVED:
                return SAVED
        if self.field_status[pos + ball_velo] == GOAL:
            self.field_status[pos + ball_velo] = BALL
            self.field_status[pos] = ROBOT
            return FAILED
        else:
            self.field_status[pos + ball_velo] = BALL
            self.field_status[pos] = FIELD
            return CONTINUE

    def result(self):
        if self.robot_state == FALL:
            return SAVED if np.random.random() < 1.0 else CONTINUE
        elif self.robot_state == STAND:
            return SAVED if np.random.random() < 0.0 else CONTINUE
        elif self.robot_state == WAKEUP:
            return CONTINUE

    def mc(self):
        self.Q = np.zeros_like(self.Q)
        num = np.zeros_like(self.Q)

        for m in self.D:
            for t in xrange(len(m)):
                d = m[t:]
                q = 0.0

                for current in xrange(len(d)):
                    q += gamma ** current * d[current][2]

                time, state, reward = m[t]
                self.Q[state, time] += q
                num[state, time] += 1.0

        num[num == 0.0] = 1.0
        self.Q / num

    def plotresult(self, states):
        for state in states:
            if state == FIELD:
                print "FIELD",
            if state == BALL:
                print "BALL",
            if state == GOAL:
                print "GOAL",
            if state == ROBOT:
                print "ROBOT",

        if self.robot_state == WAKEUP:
            s = "WAKEUP"
        elif self.robot_state == FALL:
            s = "FALL"
        else:
            s = "STAND"
        print "\tRobot state: %s" % s

if __name__ == '__main__':
    soccer = Soccer()
    for j in xrange(20):
        for i in xrange(10000):
            soccer.game()
        print "saved: %lf" % soccer.num_saved
        soccer.num_saved = 0
        soccer.mc()
        soccer.D = []

