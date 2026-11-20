class VacuumAgent:

    def __init__(self, env):
        self.environment = env
        self.loc = "A"

    def sense(self):
        return self.environment[self.loc]

    def act(self):
        status = self.sense()
        print(f"Room {self.loc} is {status}")

        if status == "Dirty":
            print(f"Sucking dirt in Room {self.loc}")
            self.environment[self.loc] = "Clean"

        if "Dirty" not in self.environment.values():
            print("Goal Achieved: All Rooms Clean")
            return True

        if self.loc == "A":
            print("Moving to Room B")
            self.loc = "B"
        elif self.loc == "B":
            print("Moving to Room A")
            self.loc = "A"

        return False

    def clean_environment(self):
        while True:
            done = self.act()
            if done:
                break


if __name__ == "__main__":
    env = {"A": "Dirty", "B": "Dirty"}

    agent = VacuumAgent(env)
    agent.clean_environment()