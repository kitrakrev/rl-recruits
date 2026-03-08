import pytest
import asyncio
from env.async_env import AsyncStaffingEnv
from env.config import Config

@pytest.mark.asyncio
async def test_async_env_step():
    config = Config(curriculum_stage=1, episode_steps=10)
    env = AsyncStaffingEnv(config)
    
    # 1. Reset
    obs, info = await env.reset(seed=42)
    assert obs.shape == (25,), "Observation shape mismatch"
    assert info["step"] == 0
    assert info["cash"] == config.seed_capital

    # 2. Step 1: get_agency_state
    obs, reward, terminated, truncated, info = await env.step({"tool": "get_agency_state"})
    assert not terminated
    assert info["step"] == 1
    
    # 3. Test concurrent calls using gather
    # We will simulate 3 different steps concurrently to ensure no deadlocks.
    # Note: In RL, environment stepping sequence matters, but we are just testing the async
    # infrastructure does not crash under typical usage.
    
    tasks = [
        env.step({"tool": "get_candidate_state"}),
        env.step({"tool": "get_client_state"}),
        env.step({"tool": "get_financial_summary"})
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All 3 steps executed
    assert len(results) == 3
    for obs, reward, terminated, truncated, info in results:
        assert obs.shape == (25,)
        assert info["step"] in (2, 3, 4) # Step counts might interleave

    assert env.core.step_count == 4
