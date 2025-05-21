import numpy as np
import pandas as pd
import statistics
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from collections import defaultdict
import json
import pandas as pd
import math
import csv
from scipy.stats import pearsonr

with open("tfidf_results_by_query.json", "r", encoding="utf-8") as file:
    tfidf_scores_dicts = json.load(file)

# convert tf-idf dicts of documents to sparse matrices
def convert_to_sparse_matrix(documents):
    unique_terms = sorted({term for doc in documents for term in doc["tfidf"].keys()})
    term_to_index = {term: idx for idx, term in enumerate(unique_terms)}

    rows, cols, data = [], [], []
    for doc_idx, doc in enumerate(documents):
        for term, tfidf_value in doc["tfidf"].items():
            rows.append(doc_idx)
            cols.append(term_to_index[term])
            data.append(tfidf_value)

    # URLS in the same order as the matrix rows
    doc_urls = [doc["url"] for doc in documents]

    tfidf_matrix = csr_matrix((data, (rows, cols)), shape=(len(documents), len(unique_terms)))
    
    return tfidf_matrix, doc_urls

# convert every document in every query to a sparse matrix by calling above function
def convert_docs_to_sparse_matrices(tfidf_scores_dicts):
    sparse_tfidf_matrices = {}
    for query in tfidf_scores_dicts:
        documents = tfidf_scores_dicts[query]
        tfidf_matrix, doc_urls = convert_to_sparse_matrix(documents)
        sparse_tfidf_matrices[query] = { 'tfidf': tfidf_matrix, 'doc_urls': doc_urls }
    return sparse_tfidf_matrices

# cluster documents using k-Means clustering
def cluster_documents(tfidf_matrix, doc_urls, seed, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    clustered_docs = defaultdict(list)
    
    for doc_idx, cluster_label in enumerate(clusters):
        clustered_docs[cluster_label].append(doc_urls[doc_idx])
    
    return clustered_docs

# cluster documents for each query using above function
def cluster_documents_per_query(sparse_tfidf_matrices, seed=42):
    query_clusters = {}
    for query in sparse_tfidf_matrices:
        tfidf_matrix = sparse_tfidf_matrices[query]['tfidf']
        doc_urls = sparse_tfidf_matrices[query]['doc_urls']
        clustered_docs = cluster_documents(tfidf_matrix, doc_urls, seed=seed, num_clusters=4)
        query_clusters[query] = clustered_docs
    return query_clusters


# calculate cumalative tfidf for terms in a cluster
def compute_cumalative_cluster_tfidf(query_clusters, tfidf_scores_dicts):
    cumalative_cluster_tfidf = {}

    for query in query_clusters:
        cumalative_cluster_tfidf[query] = {}
        
        for cluster_num in query_clusters[query]:
            cumalative_cluster_tfidf[query][cluster_num] = {}
            cluster_urls = query_clusters[query][cluster_num]
            
            for url in cluster_urls:
                entry = next((obj for obj in tfidf_scores_dicts[query] if obj["url"] == url), None)
                
                if entry is not None:
                    for term, tfidf_value in entry["tfidf"].items():
                        try:
                            cumalative_cluster_tfidf[query][cluster_num][term] += tfidf_value
                        except:
                            cumalative_cluster_tfidf[query][cluster_num][term] = tfidf_value

    return cumalative_cluster_tfidf

# compute goal text vectors for clusters in each query
def compute_goal_vectors(cumalative_cluster_tfidf, h=30):
    goal_vectors = {}

    for query in cumalative_cluster_tfidf:
        goal_vectors[query] = {}
        clusters = cumalative_cluster_tfidf[query]
        
        for cluster_num, tfidf_dict in clusters.items():
            goal_vectors[query][cluster_num] = {}
            top_terms = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)[:h]
    
            goal_vectors[query][cluster_num] = {term: score for term, score in top_terms}

    return goal_vectors

# compute final knowledge state of all users by summing the mapped document vectors
def compute_final_k_states(query_clusters, goal_vectors, query_click_by_worker):
    workers_final_knowledge_state = {}
    
    with open("tfidf_results_by_query.json", "r", encoding="utf-8") as file:
        doc_tfidf_dict = json.load(file)
        
        for worker_id, worker_data in query_click_by_worker.items():
            workers_final_knowledge_state[worker_id] = {}
            docs_mapped_tfidfs = {}
            
            for query_data in worker_data:
                query_term = query_data["query_term"]
                page_url = query_data["page_url"]
        
                # get tf-idf vector of the document at page_url
                query_tfidfs = doc_tfidf_dict[query_term]
                try:
                    doc_entry = next(x for x in query_tfidfs if x['url'] == page_url)
                except:
                    continue
                doc_tfidf = doc_entry['tfidf']
        
                # get goal-text vector of the cluster to which the document belongs
                query_term_clusters = query_clusters[query_term]
                doc_cluster_num = next(i for i in query_term_clusters if page_url in query_term_clusters[i])
                doc_cluster_goal_text = goal_vectors[query_term][doc_cluster_num]
        
                # map tf-idf document vector to the goal-text vector of the cluster
                # terms not in goal-text vector go to 0
                doc_mapped_tfidf = {}
                for term in doc_cluster_goal_text:
                    doc_mapped_tfidf[term] = doc_tfidf.get(term, 0.0)
        
                # store the newly mapped vector
                try:
                    docs_mapped_tfidfs[query_term].append(doc_mapped_tfidf)
                except:
                    docs_mapped_tfidfs[query_term] = [doc_mapped_tfidf]
        
            # sum up all the new mapped vectors to retrieve a worker knowledge state
            for query, tfidfs in docs_mapped_tfidfs.items():
                for tfidf in tfidfs:
                    for term, value in tfidf.items():
                        try:
                            workers_final_knowledge_state[worker_id][term] += value
                        except:
                            workers_final_knowledge_state[worker_id][term] = value

    return workers_final_knowledge_state

# a function for creating a combined goal text vector for each user, depending on the clusters they engaged with across queries
def combine_goal_vectors(query_clusters, goal_vectors, query_click_by_worker):
    workers_combined_goal_vector = {}

    for worker_id, worker_data in query_click_by_worker.items():
        workers_combined_goal_vector[worker_id] = {}
        goal_vector_combined = defaultdict(float)
        
        for query_data in worker_data:
            query_term = query_data["query_term"]
            page_url = query_data["page_url"]
    
            clusters_for_query = query_clusters[query_term]
            
            for cluster_num, cluster_urls in clusters_for_query.items():
                if page_url in cluster_urls:
                    goal_vector = goal_vectors[query_term][cluster_num]
                    for term, value in goal_vector.items():
                        goal_vector_combined[term] += value
    
        workers_combined_goal_vector[worker_id] = goal_vector_combined

    return workers_combined_goal_vector

# a helper function to find the number of docs clicked by a user
def find_num_docs_clicked_by_worker(worker_id, query_click_by_worker):
    worker_data = query_click_by_worker[worker_id]
    return len(worker_data)

# a function to find the number of clusters each user/worker explored
def find_clusters_explored_per_worker(query_clusters, goal_vectors, query_click_by_worker):
    clusters_explored_by_worker = {}
    
    for worker_id, worker_data in query_click_by_worker.items():
        clusters_explored_by_worker[worker_id] = {}
        
        for query_data in worker_data:
            query_term = query_data["query_term"]
            page_url = query_data["page_url"]

            clusters_for_query = query_clusters[query_term]
            
            for cluster_num, cluster_urls in clusters_for_query.items():
                if page_url in cluster_urls:
                    cluster_goal_vector = goal_vectors[query_term][cluster_num]
                    custom_cluster_key = f"{query_term}-{cluster_num}"

                    try:
                        clusters_explored_by_worker[worker_id][custom_cluster_key]["num_docs_clicked"] += 1
                    except:
                        clusters_explored_by_worker[worker_id][custom_cluster_key] = {"goal_vector": cluster_goal_vector, "num_docs_clicked": 1}

    return clusters_explored_by_worker

# function to compute the cosine similarity between two vectors (final knowledge state and goal text vectpr)
def cosine_similarity(vec1, vec2):
    # get union of keys
    terms = set(vec1) | set(vec2)
    dot = sum(vec1.get(t, 0.0) * vec2.get(t, 0.0) for t in terms)
    norm1 = math.sqrt(sum(vec1.get(t, 0.0)**2 for t in terms))
    norm2 = math.sqrt(sum(vec2.get(t, 0.0)**2 for t in terms))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# function to load up the actual knowledge gains from the dataset
def get_actual_knowledge_gains():
    actual_knowledge_gains = {}
    headers = ["worker_id", "pre_score", "post_score"]
    
    with open("./session_data/NASA_test-score.csv", mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=headers)

        for row in reader:
            worker_id = int(row["worker_id"])
            pre_score = row["pre_score"]
            post_score = row["post_score"]
            knowledge_gain = float(post_score) - float(pre_score)
    
            # ignore negative knowledge gains
            if knowledge_gain >= 0.0:
                actual_knowledge_gains[worker_id] = knowledge_gain

    return actual_knowledge_gains


###########################################################################
# Knowledge gain estimation functions (5 different methods)

# 1 Using a combined goal text vector
def estimate_k_gain_combined_goal_g_vector(final_k_states, goal_vectors, query_clusters, query_click_by_worker):
    estimated_knowledge_gains = {}
    workers_combined_goal_vector = combine_goal_vectors(query_clusters, goal_vectors, query_click_by_worker)
    
    for worker_id in final_k_states:
        final_knowledge_state = final_k_states[worker_id]
        combined_goal_vector = workers_combined_goal_vector[worker_id]
        
        similarity = cosine_similarity(final_knowledge_state, combined_goal_vector)
        estimated_knowledge_gains[worker_id] = similarity

    return estimated_knowledge_gains

# 2 Using weighted average of the goal text vectors of the engaged clusters
def estimate_k_gain_weighted_avg_g_vector(final_k_states, goal_vectors, query_clusters, query_click_by_worker):
    estimated_k_gains = {}
    clusters_explored_per_worker = find_clusters_explored_per_worker(query_clusters, goal_vectors, query_click_by_worker)
    
    for worker_id, clusters_explored in clusters_explored_per_worker.items():
        weighted_similarity_per_cluster = []
        final_k_state = final_k_states[worker_id]
        total_docs_clicked_by_user = find_num_docs_clicked_by_worker(worker_id, query_click_by_worker)
        
        for c_id in clusters_explored:
            goal_vector = clusters_explored[c_id]["goal_vector"]
            num_docs_clicked_in_cluster = clusters_explored[c_id]["num_docs_clicked"]

            similarity = cosine_similarity(final_k_state, goal_vector)
            weight = num_docs_clicked_in_cluster / total_docs_clicked_by_user

            weighted_similarity_per_cluster.append(similarity * weight)

        estimated_k_gains[worker_id] = sum(weighted_similarity_per_cluster)

    return estimated_k_gains

# 3 Using the average of the cosine similarities between the final knowledge state vector and the goal text vectors of the engaged clusters
def estimate_k_gain_per_cluster_avg(final_k_states, goal_vectors, query_clusters, query_click_by_worker):
    estimated_k_gains = {}
    
    clusters_explored_per_worker = find_clusters_explored_per_worker(query_clusters, goal_vectors, query_click_by_worker)
    
    for worker_id, clusters_explored in clusters_explored_per_worker.items():
        final_k_state = final_k_states[worker_id]
        estimated_k_gain_per_cluster = []
        
        for c_id in clusters_explored:
            goal_vector = clusters_explored[c_id]["goal_vector"]
            similarity = cosine_similarity(final_k_state, goal_vector)
            estimated_k_gain_per_cluster.append(similarity)

        estimated_k_gains[worker_id] = sum(estimated_k_gain_per_cluster) / len(estimated_k_gain_per_cluster)

    return estimated_k_gains

#4 Using the goal text vector of the dominant cluster -- cluster that was engaged with the most
def estimate_k_gain_dominant_cluster(final_k_states, goal_vectors, query_clusters, query_click_by_worker):
    estimated_k_gains = {}

    clusters_explored_per_worker = find_clusters_explored_per_worker(query_clusters, goal_vectors, query_click_by_worker)
    
    for worker_id, clusters_explored in clusters_explored_per_worker.items():
        final_k_state = final_k_states[worker_id]
        dominant_goal_vector = []
        most_clicked = 0
        
        # find dominant cluster
        for c_id in clusters_explored:
            goal_vector = clusters_explored[c_id]["goal_vector"]
            num_docs_clicked_in_cluster = clusters_explored[c_id]["num_docs_clicked"]

            if num_docs_clicked_in_cluster > most_clicked:
                most_clicked = num_docs_clicked_in_cluster
                dominant_goal_vector = goal_vector

        estimated_k_gains[worker_id] = cosine_similarity(final_k_state, dominant_goal_vector)

    return estimated_k_gains

#5 Using the median of the cosine similarities between the final knowledge state vector and the goal text vectors of the engaged clusters
def estimate_k_gain_per_cluster_median(final_k_states, goal_vectors, query_clusters, query_click_by_worker):
    estimated_k_gains = {}
    
    clusters_explored_per_worker = find_clusters_explored_per_worker(query_clusters, goal_vectors, query_click_by_worker)
    
    for worker_id, clusters_explored in clusters_explored_per_worker.items():
        final_k_state = final_k_states[worker_id]
        estimated_k_gain_per_cluster = []
        
        for c_id in clusters_explored:
            goal_vector = clusters_explored[c_id]["goal_vector"]
            similarity = cosine_similarity(final_k_state, goal_vector)
            estimated_k_gain_per_cluster.append(similarity)

        estimated_k_gains[worker_id] = statistics.median(estimated_k_gain_per_cluster)

    return estimated_k_gains

###################################################################################


# helper function to filter and then group query-and-click data by worker_id
def get_work_query_click_data(tfidf_scores_dicts):
    query_click_data = pd.read_csv("./session_data/NASA_query-and-click.csv", header=None, names=["worker_id", "query_id", "query_term", "query_time", "click_time", "serp_rank", "page_url", "page_title", "page_description"])
    filtered_data = query_click_data.dropna(subset=["page_url"])
    filtered_data = filtered_data[filtered_data["page_url"].str.strip() != ""]
    filtered_data = filtered_data[filtered_data["query_term"].isin(tfidf_scores_dicts.keys())]
    
    query_click_by_worker = filtered_data.groupby("worker_id").apply(lambda x: x.to_dict(orient="records")).to_dict()
    return query_click_by_worker

# calculate pearson correlation using k_gain from combined goal vectors
def compute_pearson(actual_k_gains_dict, estimated_k_gains_dict):
    # convert actual and estimated to equal length arrays
    actual_k_gains = []
    estimated_k_gains = []
    for worker_id in actual_k_gains_dict:
        actual_k_gain = actual_k_gains_dict[worker_id]
        try:
            estimated_k_gain = estimated_k_gains_dict[worker_id]
            actual_k_gains.append(actual_k_gain)
            estimated_k_gains.append(estimated_k_gain)
        except:
            continue
            # print(f"{worker_id} not in estimated_k_gains.")

    # calculate pearson correlation
    correlation, p = pearsonr(estimated_k_gains, actual_k_gains)
    print(f"Pearson correlation = {correlation:.3f}, p-value = {p:.4f}")

    return correlation

# function to generate random n k-Means clusterings
def get_random_initialized_doc_clusters(tfidf_sparse_matrices, n=20):
    print(f"Initializing {n} different randomly-seeded k-Means clusterings. Please wait...\n")
    clusters = []
    seeds = np.random.randint(0, 10000, size=n)

    for seed in seeds:
        query_clusters = cluster_documents_per_query(tfidf_sparse_matrices, seed=seed)
        clusters.append(query_clusters)

    return clusters

# run through the n k-Means clusters and apply the given knowledge estimation method
# then, calcualte the mean correlation as well as the standard deviation
def compute_correlation_coefficients(n_query_clusters, k_estimation_method, method_name, h, query_click_by_worker, actual_k_gains):
    coefficients = []

    print('\n', method_name, ',\th =', h)
    for q_clusters in n_query_clusters:
        cumalative_c_tfidf = compute_cumalative_cluster_tfidf(q_clusters, tfidf_scores_dicts)
        g_vects = compute_goal_vectors(cumalative_c_tfidf, h)
        fk_states = compute_final_k_states(q_clusters, g_vects, query_click_by_worker)

        # choose the knowledge estimation method here
        est_k_gains = k_estimation_method(fk_states, g_vects, q_clusters, query_click_by_worker)

        coefficient = compute_pearson(actual_k_gains, est_k_gains)
        coefficients.append(coefficient)

    mean_corr = np.mean(coefficients)
    std_corr = np.std(coefficients)
    
    print(f"\nMean correlation: {mean_corr:.4f}")
    print(f"Standard deviation: {std_corr:.4f}")
    print()

# run the algorithm
def main():
  h = 30

  # convert every document to a sparse matrix
  sparse_tfidf_matrices = convert_docs_to_sparse_matrices(tfidf_scores_dicts)
  
  # generate k-Means clusters using 20 different seeds
  n_clusters = get_random_initialized_doc_clusters(sparse_tfidf_matrices, 20)

  actual_k_gains = get_actual_knowledge_gains()
  query_click_by_worker = query_click_by_worker = get_work_query_click_data(tfidf_scores_dicts)

  # testing the 5 different knowledge estimation methods
  compute_correlation_coefficients(n_clusters, estimate_k_gain_combined_goal_g_vector, 'Combined Goal Vector', h, query_click_by_worker, actual_k_gains)
  compute_correlation_coefficients(n_clusters, estimate_k_gain_weighted_avg_g_vector, 'Weighted Vector Average', h, query_click_by_worker, actual_k_gains)
  compute_correlation_coefficients(n_clusters, estimate_k_gain_per_cluster_avg, 'Gain Per Cluster Average', h, query_click_by_worker, actual_k_gains)
  compute_correlation_coefficients(n_clusters, estimate_k_gain_dominant_cluster, 'Dominant Cluster', h, query_click_by_worker, actual_k_gains)
  compute_correlation_coefficients(n_clusters, estimate_k_gain_per_cluster_median, 'Gain Per Cluster Median', h, query_click_by_worker, actual_k_gains)
 
  # test performance of the median (5) algorithm with different h_values
  h_values = [10, 20, 30, 40, 50, 60, 70]
  for h in h_values:
    compute_correlation_coefficients(n_clusters, estimate_k_gain_per_cluster_median, 'Gain Per Cluster Median', h, query_click_by_worker, actual_k_gains)



if __name__ == "__main__":
  main()