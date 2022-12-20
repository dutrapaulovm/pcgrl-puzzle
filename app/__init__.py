from gym.envs.registration import register

register(
    id='mazecoinplay-v0',
    entry_point='mazecoinplay_env:MazeCoinPlayEnv',
    kwargs={"tile_size" : 16, "board" : (3, 2) }  
)