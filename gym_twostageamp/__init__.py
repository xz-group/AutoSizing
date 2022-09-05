from gym.envs.registration import register

register(
    id='twostageamp-v0',
    entry_point='gym_twostageamp.envs:TwoStageAmp',
)

register(
    id='twostageamp-v1',
    entry_point='gym_twostageamp.envs:TwoStageAmp_1',
)

register(
    id='twostageamp-v2',
    entry_point='gym_twostageamp.envs:TwoStageAmp_2',
)