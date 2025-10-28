# We take the TSP as an example

# Import the required classes.
import numpy as np                  # Numpy
from ml4co_kit import ATSPWrapper    # The wrapper for TSP, used to manage data and parallel generation.
from ml4co_kit import ATSPGenerator  # The generator for TSP, used to generate a single instance.
from ml4co_kit import ATSP_TYPE      # The distribution types supported by the generator.
from ml4co_kit import LKHSolver     # We choose LKHSolver to solve TSP instances
from ml4co_kit import GurobiSolver  # We choose GurobiSolver to solve TSP instances


# Check which distributions are supported by the TSP types.
for type in ATSP_TYPE:
    print(type)  # Print the supported TSP types


# Set the generator parameters according to the requirements.
atsp_generator = ATSPGenerator(
    distribution_type=ATSP_TYPE.UNIFORM,   # Generate a TSP instance with a Gaussian distribution
    precision=np.float32,                  # Floating-point precision: 32-bit
    nodes_num=40
)


atsp_solver = GurobiSolver(
    gurobi_time_limit=60,  # Time limit for Gurobi solver in seconds
)

# Create the TSP wrapper
atsp_wrapper = ATSPWrapper(precision=np.float32)

# Use ``generate_w_to_txt`` to generate a dataset of TSP.
atsp_wrapper.generate_w_to_txt(
    file_path="atsp_uniform_100ins.txt",  # Path to the output file where the generated TSP instances will be saved
    generator=atsp_generator,             # The TSP instance generator to use
    solver=atsp_solver,                   # The TSP solver to use
    num_samples=128,                      # Number of TSP instances to generate
    num_threads=1,                       # Number of CPU threads to use for parallelization; cannot both be non-1 with batch_size
    batch_size=1,                        # Batch size for parallel processing; cannot both be non-1 with num_threads
    write_per_iters=1,                   # Number of sub-generation steps after which data will be written to the file
    write_mode="a",                      # Write mode for the output file ("a" for append)
    show_time=True,                      # Whether to display the time taken for the generation process
)
