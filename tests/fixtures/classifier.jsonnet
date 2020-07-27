local USE_LAZY_DATASET_READER = std.parseInt(std.extVar("LAZY_DATASET_READER")) == 1;

// GPU to use. Setting this to -1 will mean that we'll use the CPU.
local CUDA_DEVICE = std.parseInt(std.extVar("CUDA_DEVICE"));

// Paths to data.
local TRAIN_PATH = "tests/fixtures/imdb/train.jsonl";
local DEV_PATH = "tests/fixtures/imdb/dev.jsonl";
local TEST_PATH = "tests/fixtures/imdb/test.jsonl";

// learning rate of overall model.
local LEARNING_RATE = std.extVar("LEARNING_RATE");

// dropout applied after pooling
local DROPOUT = std.extVar("DROPOUT");

local BATCH_SIZE = std.parseInt(std.extVar("BATCH_SIZE"));

local EVALUATE_ON_TEST = std.parseInt(std.extVar("EVALUATE_ON_TEST")) == 1;

local NUM_EPOCHS = std.parseInt(std.extVar("NUM_EPOCHS"));

local SEED = std.parseInt(std.extVar("SEED"));

local CNN_FIELDS(max_filter_size, embedding_dim, hidden_size, num_filters) = {
        "type": "cnn",
        "ngram_filter_sizes": std.range(1, max_filter_size),
        "num_filters": num_filters,
        "embedding_dim": embedding_dim,
        "output_dim": hidden_size, 
};


local GLOVE_FIELDS(trainable) = {
  "glove_indexer": {
    "tokens": {
      "type": "single_id",
      "lowercase_tokens": true,
    }
  },
  "glove_embedder": {
    "tokens": {
        "embedding_dim": 50,
        "trainable": trainable,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
    }
  },
  "embedding_dim": 50
};


local GLOVE_TRAINABLE = true;


local TOKEN_INDEXERS =  GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_indexer'];

local TOKEN_EMBEDDERS = GLOVE_FIELDS(GLOVE_TRAINABLE)['glove_embedder'];


local EMBEDDING_DIM = GLOVE_FIELDS(GLOVE_TRAINABLE)['embedding_dim'];

local ENCODER = CNN_FIELDS(std.parseInt(std.extVar("MAX_FILTER_SIZE")), EMBEDDING_DIM, std.parseInt(std.extVar("HIDDEN_SIZE")), std.extVar("NUM_FILTERS"));

local OUTPUT_LAYER_DIM = std.parseInt(std.extVar("HIDDEN_SIZE"));


local OUTPUT_LAYER_HIDDEN_DIM = if std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")) == 1 then [OUTPUT_LAYER_DIM] else [] + 
                                if std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")) == 2 then [OUTPUT_LAYER_DIM, OUTPUT_LAYER_DIM] else [] + 
                                if std.parseInt(std.extVar("NUM_OUTPUT_LAYERS")) == 3 then [OUTPUT_LAYER_DIM, OUTPUT_LAYER_DIM, OUTPUT_LAYER_DIM] else [];

local BASE_READER(TOKEN_INDEXERS) = {
  "lazy": false,
  "type": "text_classification_json",
  "tokenizer": {
    "word_splitter": "just_spaces",
  },
  "token_indexers": TOKEN_INDEXERS,
};

{
   "numpy_seed": SEED,
   "pytorch_seed": SEED,
   "random_seed": SEED,
   "dataset_reader": BASE_READER(TOKEN_INDEXERS),
   "validation_dataset_reader": BASE_READER(TOKEN_INDEXERS),
   "datasets_for_vocab_creation": ["train"],
   "train_data_path": TRAIN_PATH,
   "validation_data_path": DEV_PATH,
   "test_data_path": if EVALUATE_ON_TEST then TEST_PATH else null,
   "evaluate_on_test": EVALUATE_ON_TEST,
   "model": {
      "type": "basic_classifier",
      "text_field_embedder": {
        "token_embedders": TOKEN_EMBEDDERS
      },
      "seq2vec_encoder": ENCODER, 
      "dropout": DROPOUT
   },	
    "iterator": {
      "batch_size": BATCH_SIZE,
      "type": "basic"
   },

   "trainer": {
      "cuda_device": CUDA_DEVICE,
      "num_epochs": NUM_EPOCHS,
      "optimizer": {
         "lr": LEARNING_RATE,
         "type": "adam"
      },
      "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "factor": 0.5, 
          "patience": 2
      },
      "patience": 10,
      "num_serialized_models_to_keep": 1,
      "validation_metric": "+accuracy"
   }
}
