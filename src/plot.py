def update_heatmap(data, heatmap, params):
    data = data[:params['num_neurons']].reshape((params['num_neurons_perrow'], params['num_neurons_percol']))
    data = data > 0
    heatmap.set_data(data)
    #plt.draw() # cannot draw from subthraed


def gather_loop(comm, spikes_innode, heatmap, params, userinput):
    while True:
        time.sleep(0.1)
        data = comm.allgather(spikes_innode)
        data = np.hstack(data[1:])
        update_heatmap(data, heatmap, params)
        comm.Bcast(userinput, root=0)
        userinput[:] = 0


def motion(ln, ax, num_neurons_percol, userinput, event):
    x = [event.xdata]
    y = [event.ydata]

    # insert to userinput array
    if event.xdata != None and event.ydata != None:
        userinput[int(math.floor(event.ydata))*num_neurons_percol + int(math.floor(event.xdata))] = 1
        #print('debug:' + str(int(math.floor(event.xdata))) + " " + str(int(math.floor(event.ydata))) + " " + str(int(math.floor(event.xdata))*num_neurons_perrow + int(math.floor(event.ydata))))

    ln.set_data(x,y)
    plt.draw()


def mstprocess(comm, spikes_innode, params):
    plt.figure()
    fig, ax = plt.subplots()
    ln, = plt.plot([],[],'x', color='blue', markersize=10)
    heatmap = ax.imshow(np.zeros((params['num_neurons_perrow'], params['num_neurons_percol'])), cmap='hot', interpolation='nearest', alpha=0.6, vmin=0, vmax=5)
    
    userinput = np.zeros(params['num_neurons'])
    plt.connect('motion_notify_event', partial(motion, ln, ax, params['num_neurons_percol'], userinput))
    gather_thread = threading.Thread(target=gather_loop, args=(comm, spikes_innode, heatmap, params, userinput), daemon=True)
    gather_thread.start()
    plt.show()


def slvprocess(comm, spikes_innode, params):
    snn_each = snn(params)
    time_index = 0
    userinput = np.zeros(params['num_neurons'])
    delay_left = np.zeros((params['num_neurons'], params['num_neurons_each']))
    while True:
        time_index += 1
        snn_each.update_LIF(spikes_innode, userinput, time_index) 
        snn_each.update_synapse(delay_left, time_index)
    
        if time_index % 5 == 0:
            data = comm.allgather(spikes_innode)
            spike_times = np.hstack(data[1:])  # shape: (N,)
            spike_matrix = np.tile(spike_times[:, np.newaxis], (1, params['num_neurons_each']))  # shape: (N, N_each)
            delay_left = np.maximum(spike_matrix + snn_each.syn_delay - 5, 0)
            comm.Bcast(userinput, root=0)
            time_index = 0
            spikes_innode = np.zeros(params['num_neurons_each'])


if __name__ == '__main__':
    # initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    import numpy as np
    import math
    import time
    from network_params import network_params

    network_params['num_neurons_each'] = int((network_params['num_neurons'] + (size - 1) - 1 ) / (size - 1))
    network_params['num_neurons_offset'] = int(network_params['num_neurons_each'] * (rank - 1))
    spikes_innode = np.zeros(network_params['num_neurons_each'], dtype=int)

    if rank == 0:
        import matplotlib.pyplot as plt
        import matplotlib
        #matplotlib.use('TkAgg')  # バックエンドをTkAggに変更
        from functools import partial
        import threading
        mstprocess(comm, spikes_innode, network_params)
    else:
        from HopfieldSNN import snn
        slvprocess(comm, spikes_innode, network_params)
