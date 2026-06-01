from collections import defaultdict
from inverted_index.inverted_index import InvertedIndex
from scoring.scoring_options import ScoringOptions, TFWeight
import math

class ScoringResults:
    # Fields
    # idf: Dict of Float { term: idf_value, ... }
    # tf_idf: Dict of Dict of Float { doc_id: { term: tf_idf_value, ... }, ... }
    # doc_lengths: Dict of Float { doc_id: euclidean_length, ... }

    # Calculates TF-IDF (Term Frequency-Inverse Document Frequency) 
    # and Euclidean lengths (for cosine normalization) 
    # Input: Inverted Index, Total number of documents, Scoring Options (TF Weighting and TF-IDF Scheme)
    # Return: Tuple of (IDF Dict, TF-IDF Dict, Document Euclidean Lengths Dict)
    # NOTE: equation used:
    # IDF(t) = log2(N / df_t)
    def __init__(
        self,
        inverted_index: InvertedIndex, 
        total_docs: int,
        scoring_options: ScoringOptions
    ):
        self.idf = {}
        self.tf_idf = defaultdict(lambda: defaultdict(float))
        self.doc_lengths = defaultdict(float)

        for term, docs in inverted_index.inverted_index.items():
            # IDF Calculation
            df_t = len(docs)
            self.idf[term] = math.log2(total_docs / df_t)

            # TF-IDF Calculation and Document Length Accumulation
            # NOTE: Uses Raw TF
            # Accumulate the squared weight for each document's length calculation
            for doc_id, tf in docs.items():
                if scoring_options.docs_tf_weight == TFWeight.RAW_TF:
                    temp_tf_idf = tf * self.idf[term]
                elif scoring_options.docs_tf_weight == TFWeight.LOG_TF:
                    temp_tf_idf = (1 + math.log2(tf)) * self.idf[term]
                elif scoring_options.docs_tf_weight == TFWeight.BINARY_TF:
                    temp_tf_idf = 1 * self.idf[term]
                elif scoring_options.docs_tf_weight == TFWeight.AUGMENTED_TF:
                    temp_tf_idf = (0.5 + 0.5 * tf / inverted_index.max_tf) * self.idf[term]

                self.tf_idf[term][doc_id] = temp_tf_idf
                self.doc_lengths[doc_id] += temp_tf_idf ** 2
                
        # Finalize the Euclidean lengths
        for doc_id in self.doc_lengths:
            self.doc_lengths[doc_id] = math.sqrt(self.doc_lengths[doc_id])