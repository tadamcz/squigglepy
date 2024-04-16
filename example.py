import os

import squigglepy as sq
import numpy as np
import matplotlib.pyplot as plt
from squigglepy.numbers import K, M
from pprint import pprint
from squigglepy.distributions import makedist


from make_distribution.client import SciPyClient, JSONClient

token = os.environ["MAKEDISTRIBUTION_API_TOKEN"]
client = SciPyClient(JSONClient(token))

pop_of_ny_2022 = sq.to(8.1*M, 8.4*M)
pct_of_pop_w_pianos = sq.to(0.2, 1) * 0.01  # We assume there are almost no people with multiple pianos

pianos_per_piano_tuner = {
    "family": {"requested": "auto"},
    "arguments": {"quantiles": [
        {"p": 0, "x": 0},
        {"p": .10, "x": 5*K},
        {"p": .50, "x": 20*K},
        {"p": .90, "x": 50*K}
    ]}
}
pianos_per_piano_tuner = makedist(pianos_per_piano_tuner, "1d/dists/", api_client=client)
pianos_per_piano_tuner.plot()
piano_tuners_per_piano = 1 / pianos_per_piano_tuner
total_tuners_in_2022 = pop_of_ny_2022 * pct_of_pop_w_pianos * piano_tuners_per_piano
samples = total_tuners_in_2022 @ 1000  # Note: `@ 1000` is shorthand to get 1000 samples

# Get mean and SD
print('Mean: {}, SD: {}'.format(round(np.mean(samples), 2),
                                round(np.std(samples), 2)))

# Get percentiles
pprint(sq.get_percentiles(samples, digits=0))

# Histogram
plt.hist(samples, bins=200)
plt.show()

# Shorter histogram
total_tuners_in_2022.plot()