from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line

class LaneGenerator:
    """
    A class to generate and plot random points around lane markings.
    """
    def __init__(self, lane_positions: List[float], lane_length: float, lane_width: float) -> None:
        """
        Initialize the LaneGenerator object.
        ...
        """
        self.lane_positions = lane_positions
        self.lane_length = lane_length
        self.lane_width = lane_width
        
    def generate_points(self, num_points_per_side: int, num_noise_points: int) -> np.ndarray:
        """
        Generate random points along each lane and noise points between lanes.
        ...
        """
        points = []

        # Calculate the center position of each lane
        lane_centers = [lane_pos for lane_pos in self.lane_positions]
        lane_edges = [lane_pos - self.lane_width / 2 for lane_pos in lane_centers]  # Left edge of each lane

        # Generate points on each lane
        for lane_pos, lane_center, lane_edge in zip(self.lane_positions, lane_centers, lane_edges):
            # Points on each lane line
            x = np.random.uniform(lane_edge, lane_edge + self.lane_width, size=(num_points_per_side,))
            y = np.random.uniform(0, self.lane_length, size=(num_points_per_side,))

            # Add these points to the list
            points.extend(zip(x, y))

        # Generate noise points between lanes
        for i in range(len(lane_centers) - 1):
            lane_center_left = lane_centers[i]
            lane_center_right = lane_centers[i + 1]
            lane_edge_left = lane_centers[i] + self.lane_width / 2
            lane_edge_right = lane_centers[i + 1] - self.lane_width / 2
            
            # Points between two lanes
            x_noise = np.random.uniform(lane_edge_left, lane_edge_right, size=(num_noise_points,))
            y_noise = np.random.uniform(0, self.lane_length, size=(num_noise_points,))
            
            # Calculate the distance from the center of the lane
            dist_from_center_left = np.abs(x_noise - lane_center_left)
            dist_from_center_right = np.abs(x_noise - lane_center_right)
            
            # Define a function that maps distance from the center to variance
            max_std = 3  # Maximum standard deviation at the edge
            min_std = 1  # Minimum standard deviation at the center
            std_deviation_left = max_std - (max_std - min_std) * (dist_from_center_left / (lane_edge_right - lane_edge_left) / 2)
            std_deviation_right = max_std - (max_std - min_std) * (dist_from_center_right / (lane_edge_right - lane_edge_left) / 2)
            
            # Add Gaussian noise with variable standard deviation
            noise_x_gaussian_left = np.random.normal(0, std_deviation_left, size=x_noise.shape)
            noise_x_gaussian_right = np.random.normal(0, std_deviation_right, size=x_noise.shape)
            x_noise += noise_x_gaussian_left + noise_x_gaussian_right

            # Add additional uniform random noise
            noise_x_uniform = np.random.uniform(-0.5, 0.5, size=x_noise.shape)  # Uniform noise between -0.1 and 0.1
            # x_noise += noise_x_uniform

            # Clip the points to be within the image boundaries
            x_noise = np.clip(x_noise, 0, self.lane_length)

            # Add these points to the list
            points.extend(zip(x_noise, y_noise))

        # Convert the list of tuples to a NumPy array
        points_array = np.array(points).T  # Transpose to get (x, y) columns
        return points_array

    def plot_points(self, points: np.ndarray) -> None:
        """
        Plot the given 2D points on a graph.
        
        Args:
            points (np.ndarray): An array of shape (N, 2) where N is the number of points.
        """
        # Extract x and y coordinates from the points array
        print(f"points.shape: {points.shape}")
        x_points, y_points = points
        
        # Create a figure and axis
        fig, ax = plt.subplots()
        
        # Scatter plot the points
        ax.scatter(x_points, y_points)
        
        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Scatter plot of 2D points')
        
        # Set the x-axis limits based on the lane length
        ax.set_xlim(0, self.lane_length)
        
        # Set the y-axis limits based on the lane positions
        ax.set_ylim(min(self.lane_positions) - self.lane_width, max(self.lane_positions) + self.lane_width)
        
        # Draw vertical lines representing the lane positions
        for pos in self.lane_positions:
            ax.axvline(pos, color='red', linestyle='--')
        
        # Show the plot
        plt.show()

def main() -> None:
    """
    Main function to run the simulation and visualization.
    """
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

    # Plot the points
    generator.plot_points(points)

if __name__ == "__main__":
    main()