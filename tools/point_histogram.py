import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    # dataset path
    PATH = "../../../../../data/PCL_Segmentation_1000_fixed/train"

    if not os.path.exists(PATH):
        raise FileNotFoundError(f"Dataset not found at {PATH}")

    # list the filenames
    filenames = os.listdir(PATH)

    # points list
    points = []

    # iterate the dataset
    for filename in filenames:

        # get the filename
        pcl_filename = os.path.join(PATH, filename)

        # open the file
        with open(pcl_filename, "r") as f:
            lines = f.readlines()

        # get the number of points
        n_points = len(lines) - 10

        # add the count to the list
        points.append(n_points)

    # plot the histogram
    plt.hist(points, bins=20)

    # set the labels
    plt.xlabel("Number of points")
    plt.ylabel("Frequency")
    plt.title("Point cloud size histogram")

    # show the plot
    plt.show()
