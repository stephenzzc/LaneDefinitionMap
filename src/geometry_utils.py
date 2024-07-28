import random
import numpy as np

from sklearn.decomposition import PCA

from line_model_nd import LineModelND

class GeometryUtils:
    @staticmethod
    def rotate_points(points: np.ndarray, angle: float, around_center: bool = True) -> np.ndarray:
        """
        Rotate a set of 2D points by a given angle.
        
        Args:
            points (np.ndarray): Array of shape (2, N) containing x and y coordinates of points.
            angle (float): Angle in radians to rotate the points.
            around_center (bool): Whether to rotate around the center of the points. Default is True.
                
        Returns:
            np.ndarray: Rotated points array.
        """
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        
        if around_center:
            # Compute the center of the points
            center = np.mean(points, axis=1, keepdims=True)
            
            # Translate the points so that the center is at the origin
            translated_points = points - center
            
            # Rotate the translated points
            rotated_points = rotation_matrix @ translated_points
            
            # Translate the points back to their original position
            rotated_points += center

            # print(f"center: {center}, rotated_center: {np.mean(rotated_points, axis=1)}")
        else:
            # Rotate the points without translating them
            rotated_points = rotation_matrix @ points
        
        return rotated_points

    @staticmethod
    def rotate_points_around_center(points: np.ndarray, angle: float, around_center_point: np.ndarray) -> np.ndarray:
        """
        Rotate a set of 2D points by a given angle.
        
        Args:
            points (np.ndarray): Array of shape (2, N) containing x and y coordinates of points.
            angle (float): Angle in radians to rotate the points.
            around_center (bool): Whether to rotate around the center of the points. Default is True.
                
        Returns:
            np.ndarray: Rotated points array.
        """
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])


        # Translate the points so that the center is at the origin
        translated_points = points - around_center_point
        
        # Rotate the translated points
        rotated_points = rotation_matrix @ translated_points
        
        # Translate the points back to their original position
        rotated_points += around_center_point
        
        return rotated_points

    @staticmethod
    def fit_lines_with_ransac(points, min_samples=3, residual_threshold=0.3, max_trials=100):
        """
        Fit lines using the RANSAC algorithm.

        Args:
            points (np.ndarray): Array of shape (2, N) containing x and y coordinates of points.
            min_samples (int): Minimum number of samples required to fit the model.
            residual_threshold (float): Maximum distance for a data point to be classified as an inlier.
            max_trials (int): Maximum number of iterations for random sample selection.

        Returns:
            model (LineModelND): Fitted line model.
            inliers (np.ndarray): Mask array indicating which points are inliers.
        """
        x, y = points
        best_model = None
        best_inliers = None
        best_score = 0

        for _ in range(max_trials):
            # Randomly select a minimum number of points to form a line
            sample_indices = random.sample(range(len(x)), min_samples)
            sample_points = np.vstack([x[sample_indices], y[sample_indices]]).T

            # Estimate the model parameters
            model = LineModelND()
            model.estimate(sample_points)

            # Check how many points are close to the line (inliers)
            inliers = np.abs(model.residuals(np.vstack([x, y]).T)) < residual_threshold
            score = np.sum(inliers)

            if score > best_score:
                best_model = model
                best_inliers = inliers
                best_score = score

        # Refit the model using all inliers
        if best_inliers is not None:
            best_model.estimate(np.vstack([x[best_inliers], y[best_inliers]]).T)

        return best_model, best_inliers
