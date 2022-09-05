# wcao@merl.com

class Memory:
    def __init__(self):
        self.actions = []
        self.states_gcn = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states_gcn[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
