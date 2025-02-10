import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Step 1: Load the dataset
data_path = 'kddcup_data_10_percent_final.csv'  # Adjust the path if needed
data = pd.read_csv(data_path, header=None)

# Assign column names
data.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
                'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
                'num_failed_logins', 'logged_in', 'num_compromised', 
                'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 
                'num_shells', 'num_access_files', 'num_outbound_cmds', 
                'is_host_login', 'is_guest_login', 'count', 'srv_count', 
                'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 
                'label']

# Step 2: Convert categorical variables to numeric
data['protocol_type'] = data['protocol_type'].astype('category').cat.codes
data['service'] = data['service'].astype('category').cat.codes
data['flag'] = data['flag'].astype('category').cat.codes

# Step 3: Split into features (X) and target (y)
X = data.drop(columns='label')
y = data['label'].apply(lambda x: 1 if x != 'normal.' else 0)  # Binary classification (normal = 0, anomalous = 1)

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Step 4: Define the fitness function
def fitness_function(selected_features, X_train, y_train, X_test, y_test):
    # Apply selected features
    selected_indices = [i for i, bit in enumerate(selected_features) if bit == 1] #Includes an index i in the resulting list if the value (bit) at that position is equal to 1.
    if len(selected_indices) == 0:  # Avoid empty feature subsets
        return 0
    
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    # Train and evaluate a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42) #random_state will give the same model structure across different runs.
    clf.fit(X_train_sel, y_train)
    y_pred = clf.predict(X_test_sel)
    return accuracy_score(y_test, y_pred)

# Step Cluster: Form different group of particle
def form_group_of_particle(particles, num_label):
    # Apply particles
    print(num_label)
    # Step 2: Compute the Hamming distance matrix
    distance_matrix = pdist(particles, metric='hamming')
    square_distance_matrix = squareform(distance_matrix)
    print("\nHamming Distance Matrix:")
    print(square_distance_matrix)

    # Step 3: Perform hierarchical clustering
    linked = linkage(distance_matrix, method='average')  # "average" can be replaced with "single" or "complete"
    print(linked)

    # Step 4: Define the number of clusters
    num_clusters = num_label  # We want cluster according to our number of label
    clusters = fcluster(linked, t=num_clusters, criterion='maxclust')
    return clusters

# Step 5: Implement Binary Particle Swarm Optimization (BPSO)
def binary_pso(num_particles, num_features, X_train, y_train, X_test, y_test,labels, max_iter=10):
    # Initialize particles and velocities
    particles = np.random.randint(2, size=(num_particles, num_features))
    velocities = np.random.uniform(-1, 1, (num_particles, num_features))
    clusters=form_group_of_particle(particles, labels)
    print(clusters)
    # Step 5: Print clustered rows
    print("\nClusters Based on Number of Clusters:")
    for cluster_id in range(1, labels + 1):
        group = np.where(clusters == cluster_id)[0]
        print(f"Cluster {cluster_id}: Rows {group.tolist()}")

    p_best = particles.copy()
    p_best_scores = np.array([fitness_function(p, X_train, y_train, X_test, y_test) for p in particles]) #Iterates over each particle p in the particles array.
    
    # Initialize g_best as a 2D array where rows are clusters and columns are features
    g_best = np.zeros((labels, num_features), dtype=int)
    g_best_score = np.zeros(labels, dtype=float)

    # Loop over clusters to compute g_best for each cluster
    for cluster_id in range(1, labels + 1):
        # Get particles in the current cluster
        group = np.where(clusters == cluster_id)[0]
        group_size = len(group)

        if group_size > 0:  # Ensure cluster is not empty
            # Find the best particle in this cluster (highest fitness score)
            best_particle_idx = group[np.argmax(p_best_scores[group])]
            cluster_g_best = p_best[best_particle_idx]  # Best particle for this cluster
            
            # Store the best particle as g_best for the current cluster (row)
            g_best[cluster_id - 1] = cluster_g_best  # -1 because cluster_id starts from 1

    print(g_best)
    # PSO hyperparameters
    c1, c2 = 2, 2 # Acceleration coefficients
    w,  wMax,  wMin= 2, 0.9, 0.4    # Inertia weight
    Vmax=4
    
    for iteration in range(max_iter):
        w=wMax-iteration*((wMax-wMin)/max_iter)

        for i in range(num_particles):
            # Identify the cluster the particle belongs to
            cluster_id = clusters[i]

            # Find the best particle (g_best) within the same cluster
            group = np.where(clusters == cluster_id)[0]
            best_particle_idx = group[np.argmax(p_best_scores[group])]
            g_best_cluster = p_best[best_particle_idx]  # Cluster-specific g_best
            

            # Update velocity using the cluster-specific g_best
            r1, r2 = np.random.rand(num_features), np.random.rand(num_features)
            velocities[i] = (w * velocities[i] + 
                            c1 * r1 * (p_best[i] - particles[i]) +
                            c2 * r2 * (g_best_cluster - particles[i]))
            velocities[i] = np.clip(velocities[i], -Vmax, Vmax)

            # Update position using sigmoid function
            probabilities = 1 / (1 + np.exp(-2 * velocities[i]))
            particles[i] = np.where(np.random.rand(num_features) < probabilities, 1, 0)

            # Evaluate fitness
            score = fitness_function(particles[i], X_train, y_train, X_test, y_test)
            print(score)
            if score > p_best_scores[i]:
                p_best[i] = particles[i]
                g_best[cluster_id-1] = particles[i]
                p_best_scores[i] = score
                g_best_score[cluster_id-1]=score
    return g_best, g_best_score, p_best_scores

# Assuming the label column is the last column
label_column = data.columns[-1]
# Extract unique labels
labels = data[label_column].unique()
# Print the total number of labels and the labels themselves
#print(f"Total number of labels: {len(labels)}")
print("Labels of Data:", labels)

# Step 6: Run BPSO
num_particles = 100
num_features = X_train.shape[1] #Accesses the second element (num_samples, num_features) of the tuple, which corresponds to the number of features (columns) in the dataset.
g_best_list, g_best_score, best_score = binary_pso(num_particles, num_features, X_train, y_train, X_test, y_test,len(labels))
print("Best feature subset:", g_best_list)
print("G-Best score:", g_best_score)
print("Maximum P-Best score:", best_score)

# Store models and their predictions
models = []
all_predictions = []

for g_best in g_best_list:
    # Select features based on the current g_best
    selected_indices = [i for i, bit in enumerate(g_best) if bit == 1]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train_selected, y_train)
    models.append(clf)
    
    # Predict using the current model
    y_pred = clf.predict(X_test_selected)
    all_predictions.append(y_pred)

# Convert predictions to a NumPy array for easier aggregation
all_predictions = np.array(all_predictions)

# Majority voting for final prediction
# Axis=0 aggregates predictions for each test sample across all models
final_predictions = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions
)

# Evaluate the final predictions
accuracy = accuracy_score(y_test, final_predictions)
print("Final Accuracy:", accuracy)

print("Classification Report on Reduced Features:")
print(classification_report(y_test, final_predictions))

# Compute confusion matrix
cm = confusion_matrix(y_test, final_predictions)

print("Confusion Matrix:")
print(cm)
