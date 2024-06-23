import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.iloc[0]
    df = df[1:]
    # Remove the last column (label)
    df = df.iloc[:, :-1]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)
    return torch.tensor(data, dtype=torch.float32), df


def initialize_fcm(data, n_clusters, m=2):
    n_samples = data.shape[0]
    U = np.random.dirichlet(np.ones(n_clusters), size=n_samples)
    return torch.tensor(U, dtype=torch.float32)


def update_centers(data, U, m):
    Um = U ** m
    centers = (Um.T @ data) / Um.sum(axis=0)[:, None]
    return centers


class GRNN(torch.nn.Module):
    def __init__(self, sigma):
        super(GRNN, self).__init__()
        self.sigma = sigma

    def forward(self, x, centers):
        dist = torch.cdist(x, centers)
        weights = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))
        return weights / weights.sum(dim=1, keepdim=True)


def fcm_grnn_clustering(data, n_clusters, sigma=1.0, m=2, tol=1e-4, max_iter=100):
    U = initialize_fcm(data, n_clusters, m)
    centers = update_centers(data, U, m)
    grnn = GRNN(sigma)

    for _ in range(max_iter):
        U_prev = U.clone()
        weights = grnn(data, centers)
        U = weights / (weights.sum(dim=1, keepdim=True))
        centers = update_centers(data, U, m)

        if torch.norm(U - U_prev) < tol:
            break

    return U, centers


def save_clusters(data, original_df, U, n_clusters):
    cluster_assignments = torch.argmax(U, dim=1).numpy()
    for i in range(n_clusters):
        cluster_data = original_df[cluster_assignments == i]
        #cluster_data.to_csv(f'cluster_{i}.csv', index=False)
    return cluster_assignments


def select_negative_samples(original_df, cluster_assignments, sample_ratio=0.2):
    negative_samples = pd.DataFrame()
    n_clusters = len(np.unique(cluster_assignments))

    for i in range(n_clusters):
        cluster_data = original_df[cluster_assignments == i]
        sample_size = int(sample_ratio * len(cluster_data))
        sampled_data = cluster_data.sample(n=sample_size, random_state=42)
        negative_samples = pd.concat([negative_samples, sampled_data], axis=0)

    negative_samples.to_csv('./dataset/train/AAC_kcr_cvN_FCMGRNN.csv', index=False)


def main(file_path, n_clusters=3, sigma=1.0, m=2, tol=1e-4, max_iter=100):
    data, original_df = load_data(file_path)
    U, centers = fcm_grnn_clustering(data, n_clusters, sigma, m, tol, max_iter)
    cluster_assignments = save_clusters(data, original_df, U, n_clusters)

    # Select a portion of samples from each cluster as negative samples
    select_negative_samples(original_df, cluster_assignments, sample_ratio=0.2)


# Usage
main('./dataset/train/AAC_kcr_cvN.csv', n_clusters=3)