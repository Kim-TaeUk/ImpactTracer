import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

np.random.seed(42)
data = {
    'microservice': ['A', 'B', 'C', 'D', 'E', 'F'],
    'transaction_count': np.random.randint(1000, 2000, 6),
    'response_rate': np.random.uniform(0.9, 1, 6),
    'success_rate': np.random.uniform(0.9, 1, 6),
    'mean_response_time': np.random.uniform(0.1, 0.5, 6)
}

df = pd.DataFrame(data)
print("Step 1: 데이터 준비")
print(df)

bgi_features = ['transaction_count', 'response_rate', 'success_rate', 'mean_response_time']
X = df[bgi_features]

iso_forest = IsolationForest(contamination=0.1)
df['anomaly_score'] = iso_forest.fit_predict(X)
df['anomaly'] = df['anomaly_score'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

print("\nStep 2: 이상 탐지 결과")
print(df)

anomalous_nodes = df[df['anomaly'] == 'Anomaly']['microservice'].tolist()
print(f"\nAnomalous nodes: {anomalous_nodes}")

edges = [
    ('A', 'B'), ('A', 'C'), ('B', 'D'),
    ('C', 'D'), ('C', 'E'), ('D', 'F')
]

G = nx.DiGraph()
G.add_edges_from(edges)

for edge in G.edges:
    G.edges[edge]['weight'] = np.random.rand()

pos = nx.spring_layout(G)
plt.figure()
nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=14, font_weight='bold')
plt.title('Microservice Interaction Graph')
plt.show()

print("\nStep 3: 영향 그래프 구축")
print("Edges with weights:")
for edge in G.edges(data=True):
    print(edge)


def backward_trace(G, anomalies):
    reverse_G = G.reverse()
    suspicion_scores = {node: 0 for node in reverse_G.nodes}

    for node in anomalies:
        suspicion_scores[node] = 1

    for _ in range(len(reverse_G)):
        new_scores = suspicion_scores.copy()
        for node in reverse_G.nodes:
            for pred in reverse_G.predecessors(node):
                new_scores[pred] += suspicion_scores[node] * reverse_G[pred][node]['weight']
        suspicion_scores = new_scores

    return suspicion_scores


suspicion_scores = backward_trace(G, anomalous_nodes)
print("\nStep 4: 역추적 알고리즘 구현")
print("Suspicion scores:", suspicion_scores)

root_cause = max(suspicion_scores, key=suspicion_scores.get)
print(f"\nThe root cause node is: {root_cause}")
