# Test Suite

This directory contains unit tests and integration tests for the Scotland Yard game environment and agent implementations.

## Files

### `env_test.py`
**Basic environment functionality tests.** This file:
- Tests core environment operations (reset, step)
- Verifies environment doesn't crash during normal operation
- Ensures PettingZoo interface compliance
- Quick smoke tests for environment correctness

**Test Cases:**

**`test_env_reset_does_not_throw`**
- Purpose: Verify environment initializes without errors
- What it tests:
  - Environment creation succeeds
  - Reset returns valid observations
  - All agents have observations
  - Observations have correct structure
- Why important: Catches initialization bugs early

**`test_env_step_does_not_throw`**
- Purpose: Verify basic step operation works
- What it tests:
  - Environment accepts actions
  - Step returns next observations
  - Rewards are computed
  - Termination signals work
- Why important: Ensures core game loop functions

**Usage:**
```bash
pytest test/env_test.py -v
```

**What to check if tests fail:**
- Environment initialization parameters
- Observation space definitions
- Action space definitions
- Reward calculation logic
- PettingZoo interface implementation

### `test_action_mask.py`
**Action masking system tests.** This file:
- Tests valid action computation
- Verifies budget constraints are enforced
- Checks graph structure constraints
- Ensures no invalid actions are selected

**Test Cases:**

**`test_action_mask_respects_budget_and_mapping`**
- Purpose: Verify budget constraints work correctly
- Scenario: Agent with limited budget on weighted graph
- Tests:
  - Can't move to expensive neighbors
  - Can move to affordable neighbors
  - Can stay in place (always valid)
- Expected: Only affordable moves are valid

**`test_action_mask_with_scalar_toll`**
- Purpose: Handle scalar (single value) edge weights
- Scenario: All edges have same cost
- Tests:
  - Uniform edge weights processed correctly
  - Budget constraint applies uniformly
- Expected: Consistent masking with scalar weights

**`test_action_mask_fixed_index_node_mapping`**
- Purpose: Verify node indexing consistency
- Scenario: Fixed node-to-index mapping
- Tests:
  - Actions map to correct nodes
  - Index translation is consistent
  - No off-by-one errors
- Expected: Actions correspond to intended nodes

**`test_action_mask_no_valid_moves`**
- Purpose: Handle case with no valid moves except stay
- Scenario: Agent broke, can't afford any moves
- Tests:
  - Mask has no valid moves (except action 0)
  - Agent can still stay in place
  - No crash when stuck
- Expected: Only action 0 (stay) is valid

**`test_action_mask_isolated_node`**
- Purpose: Handle disconnected graph regions
- Scenario: Agent on node with no neighbors
- Tests:
  - Isolated nodes handled gracefully
  - Only stay action is valid
  - No indexing errors
- Expected: Mask correctly shows isolation

**Usage:**
```bash
pytest test/test_action_mask.py -v
```

**What these tests prevent:**
- Agents attempting invalid moves
- Budget constraint violations
- Graph structure inconsistencies
- Index out of bounds errors

### `test_belief_update.py`
**Belief system tests.** This file:
- Tests belief update mechanisms
- Verifies belief propagation on graphs
- Checks observation integration
- Ensures belief normalization

**Test Cases:**

**`test_belief_updates_and_reveals`**
- Purpose: Verify belief update logic is correct
- Scenario: Police track MrX with partial observations
- Tests multiple stages:
  
  **Stage 1: Initial Beliefs (Uniform)**
  - All nodes equally likely initially
  - Probability = 1/num_nodes for each
  - Represents complete uncertainty
  
  **Stage 2: Observation Update**
  - MrX moves to specific node
  - Police receive observation (partial reveal)
  - Belief should concentrate on revealed region
  
  **Stage 3: Belief Propagation**
  - Without new observations, beliefs diffuse
  - Neighboring nodes gain probability mass
  - Models uncertainty growth over time
  
  **Stage 4: Direct Reveal**
  - MrX position directly observed
  - Belief should spike at true position (near 1.0)
  - Other nodes should have near-zero belief

- Expected:
  - Beliefs always sum to 1.0 (probability distribution)
  - Observed positions have high belief
  - Beliefs spread to neighbors over time
  - Direct observations override propagated beliefs

**Usage:**
```bash
pytest test/test_belief_update.py -v
```

**What this tests:**
- Belief initialization (uniform distribution)
- Observation integration (Bayesian update)
- Belief propagation (diffusion on graph)
- Normalization (valid probability distribution)
- Edge cases (complete certainty, complete uncertainty)

## Running Tests

### Run All Tests
```bash
pytest test/ -v
```

### Run Specific Test File
```bash
pytest test/env_test.py -v
```

### Run Specific Test Case
```bash
pytest test/test_action_mask.py::test_action_mask_respects_budget_and_mapping -v
```

### Run with Coverage
```bash
pytest test/ --cov=src --cov-report=html
```

### Run Tests in Docker
```bash
docker run --rm --mount type=bind,src=$PWD,dst=/app student_mechanism_design pytest test/ -v
```

## Test Configuration

### `pytest.ini` (in project root)
```ini
[pytest]
testpaths = test
python_files = test_*.py *_test.py
python_functions = test_*
```

**Configuration Options:**
- `testpaths`: Where to find tests
- `python_files`: Test file naming pattern
- `python_functions`: Test function naming pattern

## Writing New Tests

### Test Structure Template
```python
import pytest
from src.environment.yard import Yard

def test_feature_description():
    """
    Test description explaining:
    - What is being tested
    - Why it matters
    - Expected behavior
    """
    # Arrange: Set up test environment
    env = Yard(config)
    env.reset()
    
    # Act: Perform the action being tested
    result = env.some_method()
    
    # Assert: Verify expected behavior
    assert result == expected_value
    assert condition_is_true
```

### Best Practices

**1. Test One Thing at a Time**
```python
# Good: Specific test
def test_agent_respects_budget():
    # Test only budget constraint
    pass

# Bad: Tests multiple things
def test_agent_does_everything():
    # Tests budget, actions, rewards, etc.
    pass
```

**2. Use Descriptive Names**
```python
# Good: Clear what's being tested
def test_action_mask_respects_budget_constraints():
    pass

# Bad: Vague
def test_action_mask():
    pass
```

**3. Include Edge Cases**
```python
def test_belief_update():
    # Normal case
    test_normal_observation()
    
    # Edge cases
    test_zero_probability()
    test_complete_certainty()
    test_empty_observation()
```

**4. Use Fixtures for Common Setup**
```python
@pytest.fixture
def env():
    """Create environment for testing."""
    config = {...}
    return Yard(config)

def test_with_fixture(env):
    # env is automatically created
    env.reset()
    assert env.num_agents > 0
```

## Test Coverage

Current coverage (as of last run):
```
test/env_test.py::test_env_reset_does_not_throw PASSED                      [ 12%]
test/env_test.py::test_env_step_does_not_throw PASSED                       [ 25%]
test/test_action_mask.py::test_action_mask_respects_budget_and_mapping PASSED [ 37%]
test/test_action_mask.py::test_action_mask_with_scalar_toll PASSED          [ 50%]
test/test_action_mask.py::test_action_mask_fixed_index_node_mapping PASSED  [ 62%]
test/test_action_mask.py::test_action_mask_no_valid_moves PASSED            [ 75%]
test/test_action_mask.py::test_action_mask_isolated_node PASSED             [ 87%]
test/test_belief_update.py::test_belief_updates_and_reveals PASSED          [100%]

8 passed, 2 warnings in 2.14s
```

**Coverage Areas:**
- ✅ Environment initialization
- ✅ Basic step operations
- ✅ Action masking (5 tests)
- ✅ Belief updates
- ❌ Reward calculations (future work)
- ❌ Agent training (future work)
- ❌ Visualization (future work)

## Debugging Failed Tests

### Test Fails: What to Do?

**1. Read the Error Message**
```
FAILED test/test_action_mask.py::test_action_mask_respects_budget_and_mapping
AssertionError: assert False == True
```
- Which test failed?
- What assertion failed?
- What were the values?

**2. Run with More Verbose Output**
```bash
pytest test/test_action_mask.py -vv
```

**3. Use Print Debugging**
```python
def test_something():
    result = compute_something()
    print(f"DEBUG: result = {result}")  # Will show in pytest output
    assert result == expected
```

**4. Run in Debugger**
```bash
pytest test/test_action_mask.py --pdb
# Drops into debugger on failure
```

**5. Check Recent Changes**
- What code did you modify?
- Could it affect this test?
- Revert changes and see if test passes

## Adding Tests for New Features

When you add new features, add tests:

**Example: Adding new reward component**

1. **Add test file**: `test/test_new_reward.py`
```python
def test_new_reward_component():
    """Test new reward calculation."""
    calc = RewardCalculator(config)
    state = create_test_state()
    
    rewards = calc.calculate_new_reward(state)
    
    assert rewards['mrx'] > 0  # MrX should get positive reward
    assert rewards['police'] < 0  # Police negative
```

2. **Run test**: `pytest test/test_new_reward.py -v`

3. **If it fails**: Debug and fix implementation

4. **If it passes**: Commit code + test together

## Tips for Students

1. **Run tests often**: After every code change
2. **Understand test failures**: Don't ignore them
3. **Write tests for bugs**: When you find a bug, write a test that catches it
4. **Test edge cases**: Empty inputs, zero values, maximum values
5. **Use tests for learning**: Read tests to understand how code works
6. **Start simple**: Write simple tests first, then more complex
7. **Test driven development**: Write test first, then implementation
8. **Keep tests fast**: Slow tests won't be run often

## Common Test Patterns

**Pattern 1: Smoke Test**
```python
def test_something_works():
    """Just verify it doesn't crash."""
    env = Yard(config)
    env.reset()
    env.step(actions)
    # If we get here, it worked
```

**Pattern 2: Value Test**
```python
def test_correct_value():
    """Verify exact output."""
    result = compute(input)
    assert result == expected_value
```

**Pattern 3: Property Test**
```python
def test_property_holds():
    """Verify property is maintained."""
    beliefs = compute_beliefs()
    assert sum(beliefs) == 1.0  # Must be valid distribution
    assert all(b >= 0 for b in beliefs)  # All non-negative
```

**Pattern 4: Error Test**
```python
def test_raises_error():
    """Verify errors are raised when expected."""
    with pytest.raises(ValueError):
        invalid_operation()
```
