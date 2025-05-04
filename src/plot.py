import matplotlib.pyplot as plt
from mpi4py import MPI
from functools import partial

def motion(ln, event):
    x = [event.xdata]
    y = [event.ydata]

    ln.set_data(x,y)
    plt.draw()

def plot(comm):
    plt.figure()
    ln, = plt.plot([],[],'x')
    
    #plt.connect('motion_notify_event', motion)
    plt.connect('motion_notify_event', partial(motion, ln))
    plt.show()

#def calculate(comm):
    # initialize neural network object

    # initialize spike list

    # the time resolution of spike shownin the figure equals to  the minimum time of syanptic delay

    # wait for actions by a user  

    


if __name__ == '__main__':
    # get rank
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        plot(comm)
    #else:
    #    calcualte(comm)
