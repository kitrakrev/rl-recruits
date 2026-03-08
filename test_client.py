"""
test.py — Run this in Terminal 2 while `uv run server` is running in Terminal 1.

Usage:
    python test.py
"""

import asyncio
import httpx

BASE_URL = "http://localhost:8000"


# ── 1. HEALTH CHECK (plain HTTP, no client) ───────────────────────────────────
def test_health():
    print("\n[1] Health check...")
    r = httpx.get(f"{BASE_URL}/health")
    print(f"    Status code : {r.status_code}")
    print(f"    Response    : {r.json()}")
    assert r.status_code == 200, "Server not healthy!"
    print("    ✅ Server is healthy")


# ── 2. RAW HTTP ───────────────────────────────────────────────────────────────
def test_raw_http():
    print("\n[2] Raw HTTP reset & step...")

    r = httpx.post(f"{BASE_URL}/reset")
    print(f"    /reset → {r.status_code} | {r.text[:300]}")

    payload = {"command": "scan", "target": "localhost"}
    r = httpx.post(f"{BASE_URL}/step", json={"action": payload})
    print(f"    /step  → {r.status_code} | {r.text[:300]}")

    r = httpx.get(f"{BASE_URL}/state")
    print(f"    /state → {r.status_code} | {r.text[:300]}")
    print("    ✅ Raw HTTP OK")


# ── 3. TYPED CLIENT ───────────────────────────────────────────────────────────
async def test_typed_client():
    print("\n[3] Typed client test...")

    try:
        from my_env.client import MyEnv
        from my_env.models import MyAction
    except ImportError as e:
        print(f"    ⚠️  Import error: {e}")
        return

    # ✅ correct pattern: async with + await
    async with MyEnv(base_url=BASE_URL) as env:

        obs = await env.reset()
        print(f"    reset()")
        print(f"      output      : {obs.output}")
        print(f"      flags_found : {obs.flags_found}")
        print(f"      done        : {obs.done}")
        print(f"      reward      : {obs.reward}")

        result = await env.step(MyAction(command="scan"))
        print(f"\n    step(command='scan')")
        print(f"      output      : {result.observation.output}")
        print(f"      flags_found : {result.observation.flags_found}")
        print(f"      reward      : {result.reward}")
        print(f"      done        : {result.done}")

        result = await env.step(MyAction(command="scan", target="localhost"))
        print(f"\n    step(command='scan', target='localhost')")
        print(f"      output      : {result.observation.output}")
        print(f"      flags_found : {result.observation.flags_found}")
        print(f"      reward      : {result.reward}")
        print(f"      done        : {result.done}")

        state = await env.state()
        print(f"\n    state()")
        print(f"      step_count  : {state.step_count}")
        print(f"      total_flags : {state.total_flags}")

    print("    ✅ Typed client OK")


# ── 4. FULL EPISODE ───────────────────────────────────────────────────────────
async def test_episode():
    print("\n[4] Full episode test...")

    try:
        from my_env.client import MyEnv
        from my_env.models import MyAction
    except ImportError as e:
        print(f"    ⚠️  Import error: {e}")
        return

    commands = [
        MyAction(command="scan",    target="localhost"),
        MyAction(command="scan",    target="target.local"),
        MyAction(command="exploit", target="web_vuln"),
        MyAction(command="exploit", target="sql_inject"),
        MyAction(command="exploit", target="priv_esc"),
    ]

    async with MyEnv(base_url=BASE_URL) as env:
        obs = await env.reset()
        print(f"    Episode started | output: {obs.output}")

        total_reward = 0.0

        for i, action in enumerate(commands):
            result = await env.step(action)
            total_reward += result.reward

            print(f"    Step {i+1:2d} | cmd={action.command:<10} "
                  f"target={action.target:<15} | "
                  f"reward={result.reward:.3f} | "
                  f"flags={result.observation.flags_found} | "
                  f"done={result.done}")

            if result.done:
                print(f"    🏁 Episode finished at step {i+1}")
                break

        state = await env.state()
        print(f"\n    Final state:")
        print(f"      step_count  : {state.step_count}")
        print(f"      total_flags : {state.total_flags}")
        print(f"      total reward: {total_reward:.3f}")

    print("    ✅ Episode test OK")


# ── 5. MULTIPLE EPISODES ──────────────────────────────────────────────────────
async def test_multiple_episodes():
    print("\n[5] Multiple episodes (reset test)...")

    try:
        from my_env.client import MyEnv
        from my_env.models import MyAction
    except ImportError as e:
        print(f"    ⚠️  Import error: {e}")
        return

    async with MyEnv(base_url=BASE_URL) as env:
        for episode in range(3):
            obs = await env.reset()
            result = await env.step(MyAction(command="scan", target="localhost"))
            state = await env.state()
            print(f"    Episode {episode+1} | "
                  f"step_count={state.step_count} | "
                  f"reward={result.reward:.3f} | "
                  f"flags={result.observation.flags_found}")

    print("    ✅ Multiple episodes OK")


# ── MAIN ──────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("  OpenEnv Server Test")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    # sync tests
    test_health()
    test_raw_http()

    # async tests
    await test_typed_client()
    await test_episode()
    await test_multiple_episodes()

    print("\n" + "=" * 60)
    print("  All tests done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())   # ← correct way to run async code