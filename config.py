# trianing - Training hyperparms
MODEL_SIZE = 'gpt2' # "Model size (gpt2, gpt2-medium)"
STORE_IN_FOLDER = 'tuned_models' # Master folder to store the model.
TRAIN_MODEL = True
TRAIN_DATA_PATH = "datasets/emotion_lyrics.csv"
NUM_TRAIN_EPOCH = 5
SAVE_EVERY_N_EPOCH = 5
TRAIN_BATCH_SIZE =2
GRADIENT_ACCMULATION_STPES = 2 # This is equivalent to batch size, if the GPU has limited memory can be used instead."

# training - Optimizer hyperparams
LEARNING_RATE = 0.0000625
ADAM_EPSILON = 1e-8 # Epsilon for Adam optimizer.
WARMUP_STEPS = 0
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0 # Max gradient norm.

# generating
LOAD_MODEL_DIR = 'tuned_models/emotion_lyrics/gpt2_20220418-2245/model_epoch_5'
GEN_BATCH = 1