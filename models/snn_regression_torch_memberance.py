import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np



class SpikingDeepGalerkinNet(nn.Module):
    def __init__(self, num_hidden_layers=4, num_hidden_units=50, T_sim=10, v_threshold=1, v_reset=0.0, device='mps', d=1):
        """
        Initialize the SpikingDeepGalerkinNet with snntorch components.

        Args:
            num_hidden_layers (int): Number of hidden layers
            num_hidden_units (int): Number of neurons per hidden layer
            T_sim (int): Number of simulation time steps
            v_threshold (float): LIF neuron firing threshold
            v_reset (float): LIF neuron reset potential
            device (str): Device for computation ('cpu' or 'cuda')
            d (int): Number of spatial dimensions
        """
        super(SpikingDeepGalerkinNet, self).__init__()
        self.T_sim = T_sim
        self.d = d
        self.num_hidden_units = num_hidden_units
        self.device = torch.device(device)
        self.layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()

        # Neuron parameters
        spike_grad = surrogate.atan()
        beta = 0.5  # Decay rate
        learning = True
        reset_mechanism = 'subtract'
        # reset_mechanism = 'zero'


        # Input layer: (t, x_1, ..., x_d) -> num_hidden_units
        self.layers.append(nn.Linear(1 + d, num_hidden_units))
        self.lif_layers.append(snn.Leaky(
            beta=torch.ones(num_hidden_units) * beta,
            threshold=torch.ones(num_hidden_units) * v_threshold,
            reset_mechanism=reset_mechanism,
            learn_beta=learning,
            learn_threshold=learning,
            spike_grad=spike_grad
        ))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
            self.lif_layers.append(snn.Leaky(
                beta=torch.ones(num_hidden_units) * beta,
                threshold=torch.ones(num_hidden_units) * v_threshold,
                reset_mechanism=reset_mechanism,
                learn_beta=learning,
                learn_threshold=learning,
                spike_grad=spike_grad
            ))

        # Output layer: num_hidden_units -> 1
        self.layers.append(nn.Linear(num_hidden_units, 1))
        self.lif_layers.append(snn.Leaky(
            beta=torch.ones(num_hidden_units) * beta,
            threshold=torch.ones(num_hidden_units) * v_threshold,
            reset_mechanism="none",
            learn_beta=learning,
            learn_threshold=learning,
            spike_grad=spike_grad
        ))

        # Initialize weights
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Print network information
        params = sum(p.numel() for p in self.parameters())
        space = 20
        print(
            f"{79 * '='}\n"
            f"{' ':<20}{'LIF (snntorch)':^39}{' ':>20}\n"
            f"{79 * '-'}\n"
            f"{'Snntorch:':<{space}}{snn.__version__}\n"
            f"{'Timesteps:':<{space}}{self.T_sim}\n"
            f"{'Parameters:':<{space}}{params}\n"
            f"{'Topology:':<{space}}\n{self}\n"
            f"{79 * '='}"
        )

    def forward(self, inputs,return_spikes=False):
        """
        Forward pass through the network.

        Args:
            t (torch.Tensor): Time tensor of shape (batch_size, 1)
            x (torch.Tensor): Spatial input tensor of shape (batch_size, d)

        Returns:
            torch.Tensor: Membrane potential of the final layer, shape (T_sim, batch_size, 1)
        """
        # Ensure inputs are on the correct device
        # t = t.to(self.device)
        # x = x.to(self.device)

        # # Concatenate t and x
        # inputs = torch.cat([t, x], dim=1)  # Shape: [batch_size, 1 + d]

        # Add time dimension: [T_sim, batch_size, 1 + d]
        inputs = inputs.unsqueeze(0).repeat(self.T_sim, 1, 1)

        # Initialize membrane potentials
        mem_states = [lif.init_leaky() for lif in self.lif_layers]

        # Initialize recordings for membrane potential
        mem_out_rec = []
        spk_rec = [[] for _ in range(len(self.layers) - 2)]

        # Forward pass over T_sim time steps
        for step in range(self.T_sim):
            x_step = inputs[step]  # Shape: [batch_size, 1 + d]
            for i, (fc, lif) in enumerate(zip(self.layers, self.lif_layers)):
           
                if i < len(self.layers) - 1:

                    x_step = fc(x_step)  # Linear transformation
                    spk, mem = lif(x_step, mem_states[i])  # Spikes and membrane potential
                    mem_states[i] = mem  # Update membrane state
                    x_step = spk  # Pass spikes to next layer
                if i < len(self.layers) - 2:
                    spk_rec[i].append(float(torch.mean(spk)))
                if i == len(self.layers) - 1: 

                    cur_out = fc(mem_states[i-1])  # Linear transformation
                    spk_out, mem = lif(cur_out, mem_states[i])
                    mem_output = torch.mean(input=mem, dim=-1)
                    mem_output = torch.unsqueeze(input=mem_output, dim=-1) #(100,1) # Record membrane potential from final LIF layer
                    mem_out_rec.append(mem_output)  # Shape: [batch_size, 1]

        # Stack membrane potentials over time
        mem_out_rec = torch.stack(mem_out_rec, dim=0) 

        self.layer_spk_rec = [np.mean(layer_spike) for layer_spike in spk_rec]


        if return_spikes:
            return torch.mean(mem_out_rec, dim=0), spk_rec
        return torch.mean(mem_out_rec, dim=0) # Shape: [T_sim, batch_size, 1]
    
    def calculate_acs_macs_ops(self):

        macs = ((1 + self.d)*(self.num_hidden_units) + (self.num_hidden_units*1))*self.T_sim

        acs = 0 

        for fr in self.layer_spk_rec:

            acs += (self.num_hidden_units**2*fr)*self.T_sim

        return acs, macs