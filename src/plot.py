def update_heatmap(data, heatmap, params):
    data = data[:params['num_neurons']].reshape((params['num_neurons_perrow'], params['num_neurons_percol']))
    heatmap.set_data(data)
    #plt.draw() # cannot draw from subthraed


def gather_loop(comm, spikes_innode, heatmap, params, userinput):
    while True:
        time.sleep(0.5)
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
    time_index = 0
    userinput = np.zeros(params['num_neurons'])
    while True:
        time_index += 1
        #spikes_innode += 1
        #if time_index in range(params['num_neurons_offset'], min(params['num_neurons_offset']+params['num_neurons_each']-1, params['num_neurons']-1)):
        #    spikes_innode[time_index - params['num_neurons_offset']] += 1

        
        for i in range(params['num_neurons_offset'], min(params['num_neurons_offset']+params['num_neurons_each']-1, params['num_neurons']-1)):
            if userinput[i] > 0:
                spikes_innode[i-params['num_neurons_offset']] = 5
            else:
                spikes_innode[i-params['num_neurons_offset']] = 0

        if time_index % 5 == 0:
            data = comm.allgather(spikes_innode)
            comm.Bcast(userinput, root=0)
            #spikes_innode = np.zeros(network_params['num_neurons_each'])


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
    spikes_innode = np.zeros(network_params['num_neurons_each'])

    if rank == 0:
        import matplotlib.pyplot as plt
        from functools import partial
        import threading
        mstprocess(comm, spikes_innode, network_params)
    else:
        slvprocess(comm, spikes_innode, network_params)
