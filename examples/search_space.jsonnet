{
    "LAZY_DATASET_READER": 0,
    "SEED": {
        "sampling strategy": "integer",
        "bounds": [0, 100000]
    },
    "DROPOUT": {
        "sampling strategy": "uniform",
        "bounds": [0, 0.5]
    },
    "LEARNING_RATE": {
        "sampling strategy": "loguniform",
        "bounds": [1e-4, 1e-1]
    },
    "CUDA_DEVICE": 0,
    "EVALUATE_ON_TEST": 0,
    "MAX_FILTER_SIZE": {
        "sampling strategy": "integer",
        "bounds": [3, 6]
    },
    "HIDDEN_SIZE": {
        "sampling strategy": "integer",
        "bounds": [64, 512]
    },
    "NUM_FILTERS": {
        "sampling strategy": "integer",
        "bounds": [64, 512]
    },
    "NUM_OUTPUT_LAYERS": {
        "sampling strategy": "choice",
        "choices": [1, 2, 3]
    },
    "SEQUENCE_LENGTH": 400,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 50,
    "VALIDATION_METRIC": "+npmi"
}