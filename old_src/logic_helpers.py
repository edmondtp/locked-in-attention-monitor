import time
from collections import deque

class AttentionStateMachine:
    def __init__(self):
        self.status = "NOT LOCKED IN"
        self.vote_score = 0
        self.VOTE_CAP = 16
        self.locked_threshold = 11
        self.unlock_threshold = 4
        self.last_not_locked = 0.0

    def update(self, raw_state, now, hard_off=False):
        if hard_off:
            self.vote_score = max(self.vote_score - 4, -self.VOTE_CAP)
            self.status = "NOT LOCKED IN"
            return self.status

        is_good = raw_state in ("FACING", "WRITING", "READING")
        
        if is_good:
            self.vote_score = min(self.vote_score + 1, self.VOTE_CAP)
        else:
            # Drop score faster if already locked in
            penalty = 3 if self.status == "LOCKED IN" else 1
            self.vote_score = max(self.vote_score - penalty, -self.VOTE_CAP)

        if self.status == "NOT LOCKED IN" and self.vote_score >= self.locked_threshold:
            self.status = "LOCKED IN"
        elif self.status == "LOCKED IN" and self.vote_score <= self.unlock_threshold:
            self.status = "NOT LOCKED IN"
            self.last_not_locked = now

        return self.status

class PostureEMA:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.val = None
    def update(self, x):
        self.val = x if self.val is None else self.alpha * x + (1 - self.alpha) * self.val
        return self.val