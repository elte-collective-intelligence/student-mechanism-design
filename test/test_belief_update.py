import os
import sys
import numpy as np

# 将 sys.path 修改放在导入之前
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
from Enviroment.belief_module import ParticleBeliefTracker


def test_belief_updates_and_reveals():
    adjacency = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    tracker = ParticleBeliefTracker(
        num_nodes=3, num_particles=20, rng=np.random.default_rng(0)
    )
    belief = tracker.update(adjacency, observation_hint=[1])
    assert np.isclose(belief.sum(), 1.0)
    # Reveal should collapse distribution
    belief = tracker.update(adjacency, reveal=2)
    assert belief.argmax() == 2
    assert np.isclose(belief.sum(), 1.0)
