import numpy as np
def motion(ln, ax, event):
    x = [event.xdata]
    y = [event.ydata]

    heatmap_data = np.random.random((100, 100))
    heatmap = ax.imshow(heatmap_data, cmap='hot', interpolation='nearest', alpha=0.6)

    ln.set_data(x,y)
    plt.draw()

#def neuralactivity(ln):
#    #spikes = np.zeros((1000, 1000))
#    ln.histgram2d(spikes)

def plot(comm):
    plt.figure()
    fig, ax = plt.subplots()
    ln, = plt.plot([],[],'x', color='blue', markersize=10)
    
    #plt.connect('motion_notify_event', motion)
    plt.connect('motion_notify_event', partial(motion, ln, ax))
    #neuralactivity(ln)
    plt.show()

#def calculate(comm):
    # initialize neural network object

    # initialize spike list

    # the time resolution of spike shownin the figure equals to  the minimum time of syanptic delay

    # wait for actions by a user  

    


if __name__ == '__main__':
    # initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    import time

    if rank == 0:
        import matplotlib.pyplot as plt
        from functools import partial
        plot(comm)
    #else:
    #    calcualte(comm)
