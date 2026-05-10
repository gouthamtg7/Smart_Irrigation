import random


class SmartIrrigationEnv:

    def __init__(self):

        # initialize environment variables

        self.soil_moisture = 1
        self.temperature = 0
        self.rainfall = 0

    def reset(self):

        """
        Starts a new episode
        Generates random initial farm conditions
        """

        self.soil_moisture = random.randint(0, 2)
        self.temperature = random.randint(0, 1)
        self.rainfall = random.randint(0, 1)

        state = (
            self.soil_moisture,
            self.temperature,
            self.rainfall
        )

        return state

    def step(self, action):

        """
        Executes one action inside environment
        """

        reward = 0
        water_used = 0

        # CASE 1:
        # Dry soil + irrigation

        if self.soil_moisture == 0 and action == 1:
            reward = 10
            water_used = 1

        # CASE 2:
        # Dry soil + no irrigation

        elif self.soil_moisture == 0 and action == 0:
            reward = -15

        # CASE 3:
        # Wet soil + irrigation

        elif self.soil_moisture == 2 and action == 1:
            reward = -10
            water_used = 1

        # CASE 4:
        # Medium soil + no irrigation

        elif self.soil_moisture == 1 and action == 0:
            reward = 5

        else:
            reward = 0
            if action == 1 :
                water_used = 1

        # simulate environmental changes

        self.soil_moisture = random.randint(0, 2)
        self.temperature = random.randint(0, 1)
        self.rainfall = random.randint(0, 1)

        next_state = (
            self.soil_moisture,
            self.temperature,
            self.rainfall
        )

        done = False

        return next_state, reward, done, water_used