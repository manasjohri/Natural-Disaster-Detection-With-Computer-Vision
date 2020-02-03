import os


DATASET_PATH = "Cyclone_Wildfire_Flood_Earthquake_Database"


CLASSES = ["Cyclone", "Earthquake", "Flood", "Wildfire","No-Disater"]


TRAIN_SPLIT = 0.75
VAL_SPLIT = 0.1
TEST_SPLIT = 0.25


MIN_LR = 1e-6
MAX_LR = 1e-4
BATCH_SIZE = 32
STEP_SIZE = 8
CLR_METHOD = "triangular"
NUM_EPOCHS = 48


MODEL_PATH = os.path.sep.join(["output", "natural_disaster.model"])


LRFIND_PLOT_PATH = os.path.sep.join(["output", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["output", "training_plot.png"])
CLR_PLOT_PATH = os.path.sep.join(["output", "clr_plot.png"])
