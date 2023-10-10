import faiss
import numpy as np
class QuerySimilaritySearch:
    """
    A class for performing similarity search on query vectors using FAISS.
    """

    def __init__(self, query_vectors:np.ndarray,index_mapping_dict:dict,
                 distance_measure:str='euclidean',num_connections_per_vertex=None,
                 ef_construction=None,ef_search=None,
                 ):
        """
        Initialize the QuerySimilaritySearch instance with query vectors.

        Args:
            query_vectors (numpy.ndarray): An array containing normalized query vectors
            index_mapping_dict (dict) : A dict storing the mapping between index and chat title
            distance_measure (str): Distance measure to use for indexing. Defaults to 'euclidean'
            num_connections_per_vertex (int): Number of nearest neighbors that each vertex will connect to.
                                              An integer value must be given when
                                              distance_measure == 'hnsw'
                                              For faster search, we can set this to the number of top k needed for downstream
            ef_construction (int): depth of layers explored during search. Defaults to None.An integer value must be given when
                                              distance_measure == 'hnsw'
            ef_search (int): # depth of layers explored during index construction
        """
        distance_measures = {
                                'euclidean': faiss.IndexFlatL2,
                                'cosine_similarity': faiss.IndexFlatIP,  # Need to normalise vectors beforehand
                                # Add more distance measures and their corresponding methods here
                                'hnsw' :faiss.IndexHNSWFlat
            
                            }
        #initialize variables
        self.query_vectors = query_vectors
        self.index_mapping_dict = index_mapping_dict
        self.num_dimensions = query_vectors.shape[1]
        self.distance_measure_used = distance_measure
        # the following variables are used only for hnsw
        self.num_connections_per_vertex = num_connections_per_vertex
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        if distance_measure == 'euclidean':
            self.index = distance_measures[self.distance_measure_used](self.num_dimensions)            
        if distance_measure == 'cosine_similarity':
            '''
            Take a sample of query vectors and check if they are L2 normalised
            '''
            sample_size = min(query_vectors.shape[0],1000)
            # sample sample_size queries randomly from query_vectors
            query_vectors_sample_indices = np.random.choice(query_vectors.shape[0], size=sample_size, replace=False)
            query_vectors_sample = query_vectors[query_vectors_sample_indices]
            # check if all vectors in the selected sample are normalised
            norm_check_flags = [QuerySimilaritySearch.is_l2_normalized(query_vector) for query_vector in query_vectors_sample]
            assert all(norm_check_flags)==True, """Input vectors are not normalised. 
                                    Perform L2 normalisation before choosing cosine_similarity"""
            self.index = distance_measures[self.distance_measure_used](self.num_dimensions)
        if distance_measure == 'hnsw':
            '''
            Take a sample of query vectors and check if they are L2 normalised
            '''
            assert all([pd.notna(num_connections_per_vertex),
                        pd.notna(ef_construction),
                        pd.notna(ef_search)]), '''
                        Please initialise QuerySimilaritySearch object with -
                            1. num_connections_per_vertex
                            2. ef_construction
                            3. ef_search
                        to run with hnsw indexing'''
            # initialize index (d == 128)
            self.index = faiss.IndexHNSWFlat(self.num_dimensions, self.num_connections_per_vertex)
            # set efConstruction and efSearch parameters
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
        
        self.index.add(query_vectors)

    @staticmethod
    def is_l2_normalized(vector):
        """
        Check if the given vector is L2 normalized.

        Args:
            vector (np.ndarray): The vector to be checked.

        Returns:
            bool: True if the vector is L2 normalized, False otherwise.
        """
        norm = np.linalg.norm(vector, ord=2)
        return np.isclose(norm, 1.0)

    
    # def index_2_vector(self, *indices):
    #     """
    #     Convert indices to the corresponding vectors.

    #     Args:
    #         indices (int or list of int): Index or list of indices of the query vectors.

    #     Returns:
    #         numpy.ndarray or list of numpy.ndarray: The query vectors corresponding to the given indices.
    #     """
    #     vectors = [self.query_vectors[idx] for idx in indices if 0 <= idx < len(self.query_vectors)]
    #     return vectors

    def find_similar_querys(self, query_vector, num_neighbors=10):
        """
        Find the indices of similar queries based on a query vector.

        Args:
            query_vector (list or numpy.ndarray): The query vector for which to find similar queries.
            num_neighbors (int): Number of similar queries to retrieve.

        Returns:
            list: A list of indices of similar queries.
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        return self.index.search(query_vector.reshape(1, -1), num_neighbors + 1)
        # _, indices = self.index.search(query_vector.reshape(1, -1), num_neighbors + 1)
        # similar_query_indices = indices[0][1:]
        # return similar_query_indices

    def find_similar_querys_range(self, query_vector, distance_threshold):
        """
        Find the indices of queries within a certain distance threshold from the query vector.

        Args:
            query_vector (list or numpy.ndarray): The query vector for which to find similar queries.
            distance_threshold (float): Maximum distance threshold for the search.

        Returns:
            list: A list of indices of similar queries within the distance threshold.
        """
        query_vector = np.array(query_vector, dtype=np.float32)
        distances, indices = self.index.range_search(query_vector.reshape(1, -1), distance_threshold)
        # valid_indices = indices[0][distances[0] > 0]
        return distances, indices