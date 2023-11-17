class Percept:
    def __init__(self, percept: tuple):
        self._step = None
        self._return = None  # this will hold the return G_t
        self._state, self._action, self._reward, self._next_state, self._done = percept

    @property
    def state(self):
        return self._state

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def next_state(self):
        return self._next_state

    @property
    def done(self):
        return self._done

    @property
    def return_(self):
        return self._return

    @return_.setter
    def return_(self, value):
        self._return = value

    def __repr__(self):
        # uses SARS - format as a convention
        direction = {0: 'left', 1: 'right'}
        return '<in {} do {} get {} -> {}> {}'.format(self.state, direction[self.action], self.reward,
                                                      self.next_state, 'done' if self.done else '')

    def __hash__(self):
        return hash((self._state, self._action, self._reward, self._next_state))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._state == other._state and self._action == other._action and self._reward == other._reward and self._next_state == other._next_state
