import torch

NO_EXPERIMENTS = 35
K_FOLDS = 2
TRAIN_SIZE = 75
TEST_SIZE = 75
TOTAL_SAMPLE = 150

MAX_EPOCHS = 10000
EPSILON = 0.0001
LEARNING_RATE = 0.5

INPUT_FEATURE = 4
INPUT_WITH_BIAS = 5
HIDDEN_NEURONS = 7
HIDDEN_WITH_BIAS = 8
OUTPUT_NEURONS = 3

MODE_DECIMALS = 6
STD_POPULATION = True

BASE_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CSV = "iris.csv"
RESULTS_DIR = "results"

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]