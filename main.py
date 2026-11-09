from src.agents.agent import Agent
from src.world.world import World
from src.tools.rng import RNGService
from src.tools.protocol import Themes



if __name__ == "__main__":
    
    rng = RNGService(42)
    world = World(rng.main)
    theme = Themes.create(rng.main)
    agents = [Agent(x, rng.sub, theme) for x in range(30)]

    world.add_agents(agents)
    world.mainloop(max_rounds=3000)
