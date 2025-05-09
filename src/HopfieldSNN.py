import numpy as np

class snn():
    def __init__(self, params):
        self.params = params

        # constants
        self.dt = 0.2
        self.tau = 20.
        self.i_strength = 1.5
        self.vrest = -65.
        self.theta = -55.
        self.r_m = 16.
        self.tau_syn = 4.
        self.ref = 1

        # initialize synapse
        self.syn_w     = np.zeros((params['num_neurons'], params['num_neurons_each']))
        self.syn_delay = np.random.randn(params['num_neurons'], params['num_neurons_each']) + 1
        self.syn_g     =  np.zeros(params['num_neurons_each'])

        # initialize stimulus
        self.i_ext = np.zeros(params['num_neurons_each'])

        # initialize membrane potential
        self.v = np.zeros(params['num_neurons_each'])

        # initialize refractory period
        self.ref_left = np.zeros(params['num_neurons_each'])

    def update_LIF(self, spikes_innode, userinput, t):
        noise = (3 * np.random.randn(self.params['num_neurons_each']) + 1) * np.sqrt(self.dt)
        self.i_ext = self.i_strength * userinput[self.params['num_neurons_offset']:self.params['num_neurons_offset']+self.params['num_neurons_each']]
        self.v += ((self.vrest - self.v + self.syn_g + self.r_m * self.i_ext) * self.dt + noise) / self.tau

        #for i in range(self.params['num_neurons_each']):
        #    if self.theta < self.v[i]:
        #        self.v[i] = self.vrest
        #        spikes_innode[i] = t

        #    if 0 < self.ref_left[i]:
        #        self.ref_left[i] -= 1
        #        self.v[i] = self.vrest

        spiking = self.v > self.theta
        self.v[spiking] = self.vrest
        spikes_innode[spiking] = t

        refractory = self.ref_left > 0
        self.ref_left[refractory] -= 1
        self.v[refractory] = self.vrest


    def update_synapse(self, delay_left, t):
        #print(np.shape(delay_left))
        transmitted = (delay_left == 1)
        transmitting = delay_left > 0
        delay_left[transmitting] -= 1

        #r = np.zeros(self.params['num_neurons_each'])
        #for post in range(self.params['num_neurons_each']):
        #    for pre in range(self.params['num_neurons']):
        #        r[post] += self.w[pre, post] * transmitted[pre]
        r = transmitted.astype(float) * self.syn_w
        #print(np.shape(r))
        self.syn_g = self.syn_g * np.exp(-self.dt/self.tau_syn) + np.sum(r, axis=0)
