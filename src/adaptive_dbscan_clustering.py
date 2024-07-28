import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from typing import Tuple, List

from lane_generator import LaneGenerator
from geometry_utils import GeometryUtils

class AdaptiveDBSCANClustering:
    def __init__(self, points: np.ndarray) -> None:
        """
        Initialize the AdaptiveDBSCANClustering class.

        Parameters:
        - points (np.ndarray): An array of shape (n_samples, n_features) representing the data points.
        """
        self.points = points

    def grid_downsample(self, grid_size: float) -> np.ndarray:
        """
        Downsample the points using grid sampling.

        Parameters:
        - grid_size (float): The size of the grid cells.

        Returns:
        - np.ndarray: Downsampled points.
        """
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)

        # Calculate the number of grid cells needed
        nx = int((x_max - x_min) / grid_size) + 1
        ny = int((y_max - y_min) / grid_size) + 1

        # Initialize an empty dictionary to hold the representative point of each grid cell
        grid_cells = {}

        # Loop through all points
        for point in self.points:
            x, y = point
            # Determine the grid cell index
            ix = int((x - x_min) / grid_size)
            iy = int((y - y_min) / grid_size)

            # If the grid cell is not yet occupied, store the point
            if (ix, iy) not in grid_cells:
                grid_cells[(ix, iy)] = point

        # Convert the dictionary values to an array
        downsampled_points = np.array(list(grid_cells.values()))

        return downsampled_points

    def find_k_distance(self, min_samples: int) -> float:
        """
        Find the distance to the kth nearest neighbor for each point.

        Parameters:
        - min_samples (int): The number of neighbors to consider.

        Returns:
        - float: The average distance to the kth nearest neighbor.
        """
        # 创建 NearestNeighbors 对象
        neigh = NearestNeighbors(n_neighbors=min_samples, algorithm='kd_tree', n_jobs=-1)
        neigh.fit(self.points)

        # 计算每个点到其第 min_samples 个最近邻的距离
        distances, _ = neigh.kneighbors(self.points)

        # 选取距离的拐点作为 eps 的候选值
        # 可以选择距离的中位数或者某个百分位数
        eps = np.mean(distances[:, -1])
        return eps

    def adaptive_dbscan_clustering(self, min_samples_range: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform adaptive DBSCAN clustering.

        Parameters:
        - min_samples_range (List[int]): Range of min_samples values to try.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Filtered points and their labels.
        """
        # 使用 find_k_distance 方法计算 eps
        min_samples = min_samples_range[0]
        eps = self.find_k_distance(min_samples)

        # 定义 eps 范围
        eps_range = np.linspace(eps * 0.9, eps * 1.1, 5)  # 可以调整范围以更好地适应数据

        best_score = -1
        best_params = (None, None)
        best_labels = None
        best_filtered_points = self.points.copy()

        for min_samples in min_samples_range:
            for eps in eps_range:
                # 使用 DBSCAN 进行聚类
                db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree', n_jobs=-1)
                db.fit(self.points)

                # 获取每个点的聚类标签
                labels = db.labels_

                # 如果只有一个聚类或没有聚类，则跳过此参数组合
                if len(set(labels)) < 2 or (-1 in set(labels) and len(set(labels)) == 2):
                    continue

                # 计算轮廓系数
                score = silhouette_score(self.points, labels)

                print(f"Evaluating min_samples={min_samples}, eps={eps}, Score: {score}")

                # 更新最佳得分和参数
                if score > best_score:
                    best_score = score
                    best_params = (min_samples, eps)
                    best_labels = labels
                    # 过滤掉噪声点
                    non_noise_indices = labels != -1
                    best_filtered_points = self.points[non_noise_indices, :]

        print(f"Best parameters: {best_params}")
        print(f"Labels length: {len(best_labels)}")
        print(f"Points shape: {self.points.shape}, Best filtered points shape: {best_filtered_points.shape}")

        return best_filtered_points, best_labels

def main():
    # Define the positions of the lanes and other parameters
    lane_positions = [3.5 * i for i in range(5)]
    lane_length = 20
    lane_width = 0.8

    # Create a LaneGenerator object
    generator = LaneGenerator(lane_positions, lane_length, lane_width)

    # Generate data points
    num_points_per_side = 15 * lane_length
    num_noise_points = 25 * lane_length
    points = generator.generate_points(num_points_per_side, num_noise_points)
    # points = points.T  # Transpose to get the correct shape (n_samples, n_features)

    center_point = np.mean(points, axis=0, keepdims=True)

    # Define the rotation angle and rotate the points
    rotation_angle = np.deg2rad(30)
    rotated_points = GeometryUtils.rotate_points_around_center(points, rotation_angle, center_point)
    rotated_points = rotated_points.T

    # Apply grid downsample
    clustering = AdaptiveDBSCANClustering(rotated_points)
    grid_size = 0.1  # Grid size
    downsampled_points = clustering.grid_downsample(grid_size)

    # 参数搜索范围, 自适应聚类
    clustering = AdaptiveDBSCANClustering(downsampled_points)
    min_samples_range = range(5, 7)
    filtered_points1, labels1 = clustering.adaptive_dbscan_clustering(min_samples_range)

    # 参数搜索范围, 自适应聚类
    clustering = AdaptiveDBSCANClustering(filtered_points1)
    min_samples_range = range(15, 18)
    filtered_points2, labels2 = clustering.adaptive_dbscan_clustering(min_samples_range)

    # 绘制结果
    plt.figure(figsize=(10, 5))

    # 绘制原始点
    plt.scatter(rotated_points[:, 0], rotated_points[:, 1], c='yellow', edgecolors='k', alpha=0.7, label='source points')

    # 绘制下采样后的点
    plt.scatter(downsampled_points[:, 0], downsampled_points[:, 1], c='blue', marker='o', s=40, edgecolors='k', alpha=0.7, label='downsampled points')

    # 绘制过滤后的点
    plt.scatter(filtered_points1[:, 0], filtered_points1[:, 1], c='green', marker='+', s=40, edgecolors='k', alpha=0.7, label='1st clustered points')

    plt.scatter(filtered_points2[:, 0], filtered_points2[:, 1], c='red', marker='x', s=40, edgecolors='k', alpha=0.7, label='2nd clustered points')

    plt.legend()
    plt.title('Adaptive DBSCAN Clustering with Grid Downsample')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    main()