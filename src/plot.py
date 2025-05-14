import numpy as np
from network_params import network_params
import threading
shared_data = np.zeros((network_params['num_neurons_perrow'], network_params['num_neurons_percol']))
lock = threading.Lock()

def update(heatmap, frame):
    with lock:
        data = shared_data.copy()
    heatmap.set_data(data > 0)
    return [heatmap]

def gather_loop(comm, spikes_innode, heatmap, params, userinput):
    while True:
        time.sleep(0.1)
        data = comm.allgather(spikes_innode)
        data = np.hstack(data[1:])
        with lock:
            shared_data[:, :] = data.reshape((params['num_neurons_perrow'], params['num_neurons_percol']))

        comm.Bcast(userinput, root=0)
        userinput[:] = 0


def motion(ln, ax, num_neurons_percol, userinput, event):
    x = [event.xdata]
    y = [event.ydata]

    if event.xdata != None and event.ydata != None:
        userinput[int(math.floor(event.ydata))*num_neurons_percol + int(math.floor(event.xdata))] = 1

    ln.set_data(x,y)
    plt.draw()


def mstprocess(comm, spikes_innode, params):
    fig, ax = plt.subplots()
    ln, = plt.plot([],[],'x', color='blue', markersize=10)
    heatmap = ax.imshow(np.zeros((params['num_neurons_perrow'], params['num_neurons_percol'])), cmap='hot', interpolation='nearest', alpha=0.6, vmin=0, vmax=5)
    
    userinput = np.zeros(params['num_neurons'])
    plt.connect('motion_notify_event', partial(motion, ln, ax, params['num_neurons_percol'], userinput))
    gather_thread = threading.Thread(target=gather_loop, args=(comm, spikes_innode, heatmap, params, userinput), daemon=True)
    ani = FuncAnimation(fig, partial(update, heatmap), interval=100, cache_frame_data=False)
    gather_thread.start()
    plt.show()


def slvprocess(comm, spikes_innode, params, pattern):
    snn_each = snn(params, pattern)
    time_index = 0
    userinput = np.zeros(params['num_neurons'])
    delay_left = np.zeros((params['num_neurons'], params['num_neurons_innode']))
    while True:
        time_index += 1
        snn_each.update_LIF(spikes_innode, userinput, time_index) 
        snn_each.update_synapse(delay_left, time_index)
    
        if time_index % 5 == 0:
            data = comm.allgather(spikes_innode)
            spike_times = np.hstack(data[1:])  # shape: (N,)
            spike_matrix = np.tile(spike_times[:, np.newaxis], (1, params['num_neurons_innode']))  # shape: (N, N_each)
            delay_left = np.maximum(spike_matrix + snn_each.syn_delay - 5, 0)
            comm.Bcast(userinput, root=0)
            time_index = 0
            spikes_innode = np.zeros(params['num_neurons_innode'])


if __name__ == '__main__':
    # initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    import math
    import time

    network_params['num_neurons_each']   = int((network_params['num_neurons'] + (size - 1) - 1 ) / (size - 1))
    network_params['num_neurons_offset'] = int(network_params['num_neurons_each'] * (rank - 1))
    network_params['num_neurons_innode'] = min(network_params['num_neurons_each'], network_params['num_neurons']-network_params['num_neurons_offset'])
    spikes_innode = np.zeros(network_params['num_neurons_innode'], dtype=int)

    if rank == 0:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import matplotlib
        from functools import partial
        import threading
        mstprocess(comm, spikes_innode, network_params)
    else:
        from HopfieldSNN import snn
        pattern = np.loadtxt('nyanko.txt')
        slvprocess(comm, spikes_innode, network_params, pattern)
