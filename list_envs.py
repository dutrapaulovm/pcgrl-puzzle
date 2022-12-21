from gym import envs
import pcgrl

print([env.id for env in envs.registry.all() if "pcgrl" in env.entry_point])