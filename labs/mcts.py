# imports
from __future__ import annotations
import numpy as np
import aigs
from aigs import State, Env
from dataclasses import dataclass
from typing import Any, List, Optional


# %% Setup
env: Env

maxDepth: int = 5

# %% Algorithms
def connect_four_minimax_heuristic(state: State, maxim: bool) -> int:
    if state.ended:
        return state.point

    # Manually prioritize columns. We choose center ones
    weights = np.array([3, 4, 5, 7, 5, 4, 3], dtype=int)

    legal = np.asarray(state.legal, dtype=int) 
    score = int(np.dot(weights, legal))

    return score if maxim else -score

# Heuristic evaluation function, specific to game
def heuristic_evaluation(state: State, maxim: bool, cfg: Any) -> int:
    match getattr(cfg, "game", None):
        case "connect_four":
            return connect_four_minimax_heuristic(state, maxim)
        case _:
            return state.point if state.ended else 0
        

def minimax(state: State, maxim: bool, depth: int, cfg: Any) -> int:
    if depth > maxDepth:
        # Heuristic evaluation
        return heuristic_evaluation(state, maxim, cfg)
    if state.ended:
        return state.point
    else:
        actions = np.where(state.legal)[0]
        if actions.size == 0:
            # No legal actions: treat as terminal for robustness
            return state.point

        best = -np.inf if maxim else np.inf
        for action in actions:  # for all legal actions
            value = minimax(env.step(state, action), not maxim, depth + 1, cfg)
            if maxim:
                if value > best:
                    best = value
            else:
                if value < best:
                    best = value
        # best will be finite after at least one child
        return int(best)


def alpha_beta(state: State, maxim: bool, alpha: int, beta: int, depth: int, cfg: Any) -> int:
    if depth > maxDepth:
        # Heuristic evaluation
        return heuristic_evaluation(state, maxim, cfg)
    if state.ended:
        return state.point
    if maxim:
        actions = np.where(state.legal)[0]
        if actions.size == 0:
            return state.point
        value = -np.inf
        for action in actions:
            # From neighboring branches, what is the best value so far?
            value = max(value, alpha_beta(env.step(state, action), False, alpha, beta, depth + 1, cfg))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return int(value)
    else:
        actions = np.where(state.legal)[0]
        if actions.size == 0:
            return state.point
        value = np.inf
        for action in actions:
            # From neighboring branches, what is the best value so far?
            value = min(value, alpha_beta(env.step(state, action), True, alpha, beta, depth + 1, cfg))
            beta = min(beta, value)
            if beta <= alpha:
                break
        return int(value)


@dataclass
class Node:
    state: State                        # game state at this node
    parent: Optional["Node"] = None     # parent pointer
    action: Optional[int] = None        # action taken from parent to reach this node
    children: List["Node"] = None       # list of child nodes
    untried_actions: List[int] = None   # actions not yet expanded
    visits: int = 0                     # N(s)
    value: float = 0.0                  # W(s): cumulative reward from this node player's perspective

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            # legal actions from this state
            self.untried_actions = [int(i) for i in np.where(self.state.legal)[0]]

    @property
    def terminal(self) -> bool:
        return bool(self.state.ended)

    @property
    def fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

# Intuitive but difficult in terms of code
def monte_carlo(state: State, cfg) -> int:
    # Handle degenerate cases
    actions = np.where(state.legal)[0]
    if state.ended or actions.size == 0:
        return -1

    # Build root
    root = Node(state=state)

    # Compute budget (iterations)
    compute = int(getattr(cfg, "compute", 200))
    c = float(getattr(cfg, "c", np.sqrt(2)))

    for _ in range(max(compute, 1)):
        # Selection + Expansion
        v = tree_policy(root, c)
        # Simulation
        delta = default_policy(v.state)
        # Backpropagation
        backup(v, delta)

    # Choose the action with the highest average value from the root
    if not root.children:
        # If nothing expanded, just pick random legal
        return int(np.random.choice(actions))

    # average value from root perspective is just child.value/child.visits at child (child is from its own perspective).
    # but backup flipped perspectives so child.value is from child's player perspective.
    # To evaluate from root perspective, negate because child's player is the opponent of root.
    # However, the standard choice is to pick the child with highest visit count or highest mean from root's perspective.
    # We'll use visit count for robustness.
    best = max(root.children, key=lambda n: n.visits)
    return int(best.action) if best.action is not None else int(np.random.choice(actions))

def tree_policy(node: Node, c: float) -> Node:
    v = node
    while not v.terminal:
        if not v.fully_expanded:
            return expand(v)
        else:
            v = best_child(v, c)
            if v is None:
                break
    return v

def expand(v: Node) -> Node:
    # Select one untried action uniformly
    a = v.untried_actions.pop()  # remove one action
    next_state = env.step(v.state, a)
    child = Node(state=next_state, parent=v, action=int(a))
    v.children.append(child)
    return child

def best_child(root: Node, c: float) -> Optional[Node]:
    # UCT: argmax_i (Q_i/N_i) + c * sqrt(2 ln N / N_i)
    if not root.children:
        return None
    # Compute exploration term denominator once for efficiency and guard against log(0)
    lnN = np.log(max(root.visits, 1))
    best_score = -np.inf
    best_nodes: List[Node] = []
    for child in root.children:
        if child.visits == 0:
            uct = np.inf
        else:
            q = child.value / child.visits
            # child.value is stored from the child's player perspective; negate if players swap
            if child.state.maxim != root.state.maxim:
                q = -q
            # Combine exploitation (mean value) with exploration bonus
            uct = q + c * np.sqrt(2 * lnN / child.visits)
        if uct > best_score:
            best_score = uct
            best_nodes = [child]
        elif uct == best_score:
            best_nodes.append(child)
    return np.random.choice(best_nodes)

def default_policy(state: State) -> int:
    # Rollout to terminal with uniform random actions
    s = state
    # The reward is from the perspective of the player to act at 'state'
    player_is_maxim = s.maxim
    while not s.ended:
        actions = np.where(s.legal)[0]
        if actions.size == 0:
            break
        a = int(np.random.choice(actions))
        s = env.step(s, a)
    # At terminal, point is +1 if maxim won, -1 if minim won, 0 draw
    return int(s.point if player_is_maxim else -s.point)

def backup(node: Node, delta: int) -> None:
    # Propagate the simulation result up to the root, flipping perspective each level
    v = node
    reward = float(delta)
    while v is not None:
        v.visits += 1
        v.value += reward
        parent = v.parent
        if parent is None:
            break
        # Flip reward only when the player to move changes between parent and child
        if parent.state.maxim != v.state.maxim:
            reward = -reward
        v = parent


# Main function
def main(cfg) -> None:
    global env
    env = aigs.make(cfg.game)
    state = env.init()

    while not state.ended:
        actions = np.where(state.legal)[0]  # the actions to choose from

        match getattr(cfg, state.player):
            case "random":
                a = np.random.choice(actions).item()

            case "human":
                print(state, end="\n\n")
                a = int(input(f"Place your piece ({'x' if state.minim else 'o'}): "))

            case "minimax":
                values = [minimax(env.step(state, a), not state.maxim, 1, cfg) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "alpha_beta":
                values = [alpha_beta(env.step(state, a), not state.maxim, -np.inf, np.inf, 1, cfg) for a in actions]
                a = actions[np.argmax(values) if state.maxim else np.argmin(values)]

            case "monte_carlo":
                a = monte_carlo(state, cfg)

            case _:
                raise ValueError(f"Unknown player {state.player}")

        state = env.step(state, a)

    print(f"{['nobody', 'o', 'x'][state.point]} won", state, sep="\n")
