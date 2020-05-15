
import matplotlib.pyplot as plt

# matplotlib.use('MacOSX')


class BaseGraph:
    def __init__(self, vectors_3dim):
        self.vectors_3dim = vectors_3dim

    def getPlot(self):
        # Create a scatter plot
        fig = plt.figure(dpi=120, figsize=(8.0, 5.0))
        ax = fig.add_subplot( projection='3d')

        # drow all the nodes in the grapg
        ax.scatter(self.vectors_3dim[:, 0], self.vectors_3dim[:, 1], self.vectors_3dim[:, 2], s=1)

        # the axis labels
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        return plt

