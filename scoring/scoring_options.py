from enum import Enum

class TFWeight(Enum):
    RAW_TF = 1
    LOG_TF = 2
    BINARY_TF = 3
    AUGMENTED_TF = 4

class TFIDFScheme(Enum):
    RAW = 1
    NORMALIZED = 2

class ScoringOptions:
    def __init__(
            self, 
            docs_tf_weight: TFWeight = TFWeight.RAW_TF, 
            docs_tf_idf_scheme: TFIDFScheme = TFIDFScheme.RAW,
            query_tf_weight: TFWeight = TFWeight.RAW_TF,
            query_tf_idf_scheme: TFIDFScheme = TFIDFScheme.RAW
            ):
        self.docs_tf_weight = docs_tf_weight
        self.docs_tf_idf_scheme = docs_tf_idf_scheme
        self.query_tf_weight = query_tf_weight
        self.query_tf_idf_scheme = query_tf_idf_scheme