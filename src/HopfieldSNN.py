import numpy as np

class snn():
    def __init__(self, params, pattern):
        self.params = params

        # constants
        self.dt = 0.2
        self.tau = 20.
        self.i_strength = 15
        self.vrest = -65.
        self.theta = -55.
        self.r_m = 16.
        self.tau_syn = 4.
        self.ref = 1

        # initialize synapse
        pattern = pattern.reshape(params['num_neurons'])
        self.syn_w     =  np.zeros((params['num_neurons'], params['num_neurons_innode']))
        for pre in range(params['num_neurons']):
            for post in range(params['num_neurons_innode']):
                self.syn_w[pre, post] = pattern[pre] * pattern[post+params['num_neurons_offset']] / 10
        self.syn_delay = np.floor((np.random.randn(params['num_neurons'], params['num_neurons_innode']) + 1) / self.dt).astype(int)
        self.syn_g     =  np.zeros(params['num_neurons_innode'])

        # initialize stimulus
        self.i_ext = np.zeros(params['num_neurons_innode'])

        # initialize membrane potential
        self.v = np.full(params['num_neurons_innode'], self.vrest)

        # initialize refractory period
        self.ref_left = np.zeros(params['num_neurons_innode'])

    def update_LIF(self, spikes_innode, userinput, t):
        noise = (3 * np.random.randn(self.params['num_neurons_innode']) + 1) * np.sqrt(self.dt)
        self.i_ext = self.i_strength * userinput[self.params['num_neurons_offset']:
                                                 self.params['num_neurons_offset']+self.params['num_neurons_each']]
        self.v += ((self.vrest - self.v + self.syn_g + self.r_m * self.i_ext) * self.dt + noise) / self.tau

        spiking = self.v > self.theta
        self.v[spiking] = self.vrest
        self.ref_left[spiking] = self.ref
        spikes_innode[spiking] = t

        refractory = self.ref_left > 0
        self.ref_left[refractory] -= 1
        self.v[refractory] = self.vrest


    def update_synapse(self, delay_left, t):
        transmitted = (delay_left == 1)
        print(np.sum(transmitted))
        transmitting = delay_left > 0
        delay_left[transmitting] -= 1
        r = transmitted.astype(float) * self.syn_w
        self.syn_g = self.syn_g * np.exp(-self.dt/self.tau_syn) + np.sum(r, axis=0)
