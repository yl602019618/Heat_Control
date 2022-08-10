from gym.envs.registration import register

register(
		id='Heat_d-v0',
		entry_point='Heat.envs:Heat_d_Env',
)