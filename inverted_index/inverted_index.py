from collections import defaultdict

class InvertedIndex:
    # Fields:
    # inverted_index: Dict of Dict of Integer { term: { doc_id: tf, ... }, ... }
    # max_tf: Integer (Maximum term frequency across all documents, used for Augmented TF

    # Creates Inverted Index and Max TF Calculation
    # Input: Processed documents in the form of Dict of List of String { doc_id: [token1, token2, ...] }
    # NOTE: Assumes text is already processed into tokens 
    # Return: Inverted Index in the form of Dict of Dict of Integer
    # i.e. { term: { doc_id: tf, ... }, ... }
    # NOTE: Contains Raw TF
    def __init__(self, processed_docs: dict[str, list[str]]):
        # Initialization
        inverted_index = defaultdict(lambda: defaultdict(int))
        tf_dict = defaultdict(int)
        max_tf = 0
        
        # Populate the inverted index with raw TF 
        for doc_id, tokens in processed_docs.items():
            for token in tokens:
                inverted_index[token][doc_id] += 1
                tf_dict[token] += 1

                if (tf_dict[token] > max_tf):
                    max_tf = tf_dict[token]
                
        self.inverted_index = inverted_index
        self.max_tf = max_tf

