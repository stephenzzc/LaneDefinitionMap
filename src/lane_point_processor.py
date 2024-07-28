import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from geometry_utils import GeometryUtils

import matplotlib.pyplot as plt

class LanePointProcessor:
    def __init__(self, expected_lane_width: float, line_width: float = 1.5):
        self.expected_lane_width = expected_lane_width
        self.line_width = line_width

    def find_peaks(self, sorted_x_coordinates: np.ndarray):
        density = gaussian_kde(sorted_x_coordinates)
        density.covariance_factor = lambda : 0.1
        density._compute_covariance()

        # Find the peaks in the density function
        x_grid = np.linspace(min(sorted_x_coordinates), max(sorted_x_coordinates), 50)
        density_estimate = density(x_grid)
        peaks, _ = find_peaks(density_estimate, height=np.percentile(density_estimate, 30))
        peaks = x_grid[peaks]

        print(f"peaks: {peaks}")

        return peaks

    def find_adjacent_peaks(self, peaks, sorted_x_coordinates: np.ndarray):
        adjacent_peaks = []
        for peak in peaks:
            # Search in the range of the lane width
            search_range_left = np.linspace(peak - self.expected_lane_width * 1.5, peak - self.expected_lane_width * 0.5, 100)
            search_range_right = np.linspace(peak + self.expected_lane_width * 0.5, peak + self.expected_lane_width * 1.5, 100)

            # Check if there is already a peak within the left or right search ranges
            peaks_in_left_range = np.isin(peaks, search_range_left)
            peaks_in_right_range = np.isin(peaks, search_range_right)

            # If no peak is found in the left range, process the left side
            if not np.any(peaks_in_left_range):
                density_left = gaussian_kde(sorted_x_coordinates)
                density_left.covariance_factor = lambda : 0.1
                density_left._compute_covariance()
                density_estimate_left = density_left(search_range_left)
                new_peaks_left, _ = find_peaks(density_estimate_left, height=np.percentile(density_estimate_left, 20))
                new_peaks_left = search_range_left[new_peaks_left]
                
                # Exclude the original peak and any previously found peaks
                new_peaks_left = new_peaks_left[new_peaks_left != peak]
                new_peaks_left = new_peaks_left[np.isin(new_peaks_left, peaks, invert=True)]
                
                adjacent_peaks.extend(new_peaks_left)

            # If no peak is found in the right range, process the right side
            if not np.any(peaks_in_right_range):
                density_right = gaussian_kde(sorted_x_coordinates)
                density_right.covariance_factor = lambda : 0.1
                density_right._compute_covariance()
                density_estimate_right = density_right(search_range_right)
                new_peaks_right, _ = find_peaks(density_estimate_right, height=np.percentile(density_estimate_right, 20))
                new_peaks_right = search_range_right[new_peaks_right]
                
                # Exclude the original peak and any previously found peaks
                new_peaks_right = new_peaks_right[new_peaks_right != peak]
                new_peaks_right = new_peaks_right[np.isin(new_peaks_right, peaks, invert=True)]
                
                adjacent_peaks.extend(new_peaks_right)

        # Combine the initial peaks with the adjacent peaks
        all_peaks = np.concatenate((peaks, np.array(adjacent_peaks)))

        # Remove duplicates within the expected lane width
        unique_peaks, counts = np.unique(all_peaks, return_counts=True)
        all_peaks = unique_peaks[counts == 1]

        # Remove peaks that are too close together
        all_peaks = self.remove_close_peaks(all_peaks, self.expected_lane_width/2)

        print(f"all peaks: {all_peaks}")

        return all_peaks

    def remove_close_peaks(self, peaks, distance_threshold):
        # Sort the peaks
        sorted_peaks = np.sort(peaks)
        
        # Compute the differences between consecutive peaks
        peak_differences = np.diff(sorted_peaks)
        
        # Identify peaks that are closer than the threshold
        close_peaks = np.where(peak_differences < distance_threshold)[0] + 1
        
        # Remove the peaks that are too close to each other
        valid_peaks = np.delete(sorted_peaks, close_peaks)
        
        return valid_peaks

    def filter_points(self, vertical_points: np.ndarray, filter_points: np.ndarray, estimated_lane_positions: np.ndarray):
        # Ensure the shapes of vertical_points and filter_points match
        if vertical_points.shape != filter_points.shape:
            raise ValueError("The shapes of vertical_points and filter_points must be the same.")

        # Unpack the points
        vertical_x, vertical_y = vertical_points
        filter_x, filter_y = filter_points

        filtered_x = []
        filtered_y = []

        for pos in estimated_lane_positions:
            lane_mask = (vertical_x > pos - self.line_width / 2) & (vertical_x < pos + self.line_width / 2)
            filtered_x.extend(filter_x[lane_mask])
            filtered_y.extend(filter_y[lane_mask])

        return np.array(filtered_x), np.array(filtered_y)


if __name__ == "__main__":
    # Re-generate the sample data and rotate it
    np.random.seed(0)  # For reproducibility

    # Define the lane lengths and widths
    lane_lengths = [25, 5, 15, 15]
    expected_lane_width = 3.5
    lane_widths = [-expected_lane_width*0.5, expected_lane_width*0.5, -expected_lane_width * 1.5, expected_lane_width * 1.5]

    # Set point density per meter
    point_density_per_meter = 30

    # Generate random points along multiple lanes and rotate them
    # Initialize an empty list to store all points
    all_points = []

    for i, (length, width) in enumerate(zip(lane_lengths, lane_widths)):
        # Calculate the number of points based on the length of the lane
        n_points = int(length * point_density_per_meter)
        
        # Generate x coordinates with a normal distribution around the lane's width
        x = np.random.normal(loc=width, scale=0.8, size=n_points)
        
        # Generate y coordinates for the lane
        y = np.random.rand(n_points) * length
        
        # Combine the x and y coordinates into (x, y) pairs
        lane_points = np.column_stack((x, y))
        
        # Append the lane's points to the all_points list
        all_points.append(lane_points)

    # Combine all lanes into one array
    points = np.vstack(all_points).T

    center_point = np.mean(points, axis=1, keepdims=True)

    # Define the rotation angle and rotate the points
    rotation_angle = np.deg2rad(30)
    rotated_points = GeometryUtils.rotate_points_around_center(points, rotation_angle, center_point)

    # Create an instance of the class
    processor = LanePointProcessor(expected_lane_width)

    # Estimate the direction of the points and rotate them back
    # Calculate the angle of the fitted line
    model_robust, inliers = GeometryUtils.fit_lines_with_ransac(rotated_points)
    angle = np.arctan2(-1, model_robust.get_params()[0])
    if angle < 0:
        angle += np.pi
    print(f"line fit angle: {np.rad2deg(angle)}, model_info: {model_robust.get_params()}")

    # Rotate points back to their original orientation
    points_rotated_back = GeometryUtils.rotate_points_around_center(rotated_points, -angle, center_point)

    # Sort the x coordinates
    x, y = points_rotated_back
    sorted_indices = np.argsort(x)
    sorted_x_rotated_back_coordinates = x[sorted_indices]
    sorted_y_rotated_back_coordinates = y[sorted_indices]

    # Find the peaks in the density function
    initial_peaks = processor.find_peaks(sorted_x_rotated_back_coordinates)
    
    # Find adjacent peaks
    all_peaks = processor.find_adjacent_peaks(initial_peaks, sorted_x_rotated_back_coordinates)
    print(f"all peaks: {all_peaks}")

    # Filter the rotated back points
    filtered_points = processor.filter_points(points_rotated_back, rotated_points, all_peaks)

    # Rotate the filtered points back to the rotated coordinate system
    filtered_points_rotated_back = GeometryUtils.rotate_points_around_center(np.vstack(filtered_points), -angle, center_point)
    print(f"filtered_points shape: {np.array(filtered_points).shape}, center_point shape: {center_point.shape}")
    
    # Extract the inlier points for the fitted line
    x_inliers, y_inliers = rotated_points[:, inliers]
    line_points = model_robust.predict_xy(np.vstack([x_inliers, y_inliers]).T)

    # 绘制点云和拟合直线
    plt.figure(figsize=(10, 5))  # 增加高度以适应两个子图

    # 第一个子图
    plt.subplot(1, 2, 1)  # 第一个子图
    plt.scatter(*rotated_points, color='blue', label='Rotated Points')
    plt.scatter(*filtered_points, color='orange', alpha = 0.3, label='Filtered Points')
    plt.plot(line_points[:, 0], line_points[:, 1], color='green', label='Fitted Line')
    plt.title('Scatter plot of rotated and filtered points')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()

    # 第二个子图
    plt.subplot(1, 2, 2)  # 第二个子图
    plt.scatter(*points_rotated_back, color='yellow', label='Rotated Back Points')
    plt.scatter(*filtered_points_rotated_back, color='cyan', label='Filtered Rotated Back Points')
    plt.title('Scatter plot of rotated back and filtered points')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()


    # Set the y-axis limits based on the lane length
    plt.ylim(0, np.max(lane_lengths))

    # Set the x-axis limits based on the filtered points
    plt.xlim(min(rotated_points[0]) - expected_lane_width, max(rotated_points[0]) + expected_lane_width)

    plt.tight_layout()  # 自动调整子图布局，防止重叠
    plt.show()