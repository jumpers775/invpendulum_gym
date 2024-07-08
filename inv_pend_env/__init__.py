from gymnasium.envs.registration import register

register(
     id="inv_pend_env/inv_pendulum_v0",
     entry_point="inv_pend_env.envs.inv_pendulum_v0:InvPend",
     max_episode_steps=50,
)