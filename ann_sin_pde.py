import torch
import torch.nn as nn
import numpy as np
import configparser
import importlib
import os
from itertools import product
import pandas as pd
from pde_class import SineGordon1





def parse_list(string_list):
    return [eval(x.strip()) for x in string_list.strip('[]').split(',')]

def load_config_combinations(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    production_models = config['default']['production'].split(',')
    
    combinations = []
    for model in production_models:
        model = model.strip()
        params = {}
        for key in ['hidden_layers', 'num_hidden_units', 'T_sim', 'dimension']:
            params[key] = parse_list(config[model][key])
        
        # Generate all combinations using product
        param_combinations = list(product(*params.values()))
        param_names = params.keys()
        
        # Create a list of dictionaries for each combination
        for combo in param_combinations:
            config_combo = dict(zip(param_names, combo))
            config_combo['model'] = model
            combinations.append(config_combo)
    
    return combinations

# Computes initial loss: 𝔼[|v(0,X) - φ(X)|²]
def dgm_initial_loss(v, φ, x, dev):
    t = torch.zeros(x.shape[0],1).to(dev)
    u = v(torch.cat((t,x), axis=1))
    return (u - φ(x)).square().mean()

# Computes dynamic loss: 𝔼[|∂v/∂t(T,X) - κΔₓv(T,X) - f(v(T,X))|²]
def dgm_dynamic_loss(v, ρ, f, t, x, dev):
    d = x.shape[1]
    
    # Split x up into components
    x_comp = [x[:,i] for i in range(d)]
    
    # Compute v(t,x)
    u = v(torch.stack([t] + x_comp, 1))
    
    # Compute ∂v/∂t(t,x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    
    # Compute Δₓv(t,x)
    Δu = torch.zeros(x.shape[0]).to(dev)
    for i in range(d):
        # Compute ∂v/∂xᵢ(x,t)
        u_xi = torch.autograd.grad(u, x_comp[i], torch.ones_like(u), 
                                   create_graph=True)[0]
        # Compute ∂²v/∂xᵢ²(x,t)
        u_xixi = torch.autograd.grad(u_xi, x_comp[i], torch.ones_like(u_xi), 
                                     create_graph=True)[0]
        Δu += u_xixi
        
    return (u_t - ρ * Δu - f(u)).square().mean()

# Xavier initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.bias, std=0.1)

# Performance reporting function
def report_performance(model, φ, T, d, dev, true_value, output_folder):
    model.eval()
    with torch.no_grad():
        # Evaluate at u(T, 0, ..., 0)
        t = torch.tensor([[T]], dtype=torch.float32).to(dev)
        x = torch.zeros(1, d).to(dev)
        inputs = torch.cat((t, x), dim=1)
        predicted = model(inputs).item()
        
        # Calculate errors
        abs_l1_error = abs(predicted - true_value)
        rel_l1_error = abs_l1_error / abs(true_value) if true_value != 0 else float('inf')
        
        # Save to file
        os.makedirs(output_folder, exist_ok=True)
        report_file = os.path.join(output_folder, f'performance_d{d}.txt')
        with open(report_file, 'w') as f:
            f.write(f"Predicted u({T}, 0, ..., 0): {predicted:.6f}\n")
            f.write(f"True u({T}, 0, ..., 0): {true_value:.6f}\n")
            f.write(f"Absolute L1 Error: {abs_l1_error:.6f}\n")
            f.write(f"Relative L1 Error: {rel_l1_error:.6f}\n")
        
        print(f"Performance report saved to {report_file}")
        print(f"Absolute L1 Error: {abs_l1_error:.6f}")
        print(f"Relative L1 Error: {rel_l1_error:.6f}")
    
    return abs_l1_error, rel_l1_error




def train(pde, model, T, d, steps=1000, batch_size=100, lr=0.001, stddev=lambda t: 1.0, lr_decay=0.999, save_path=None, device='cpu'):
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    
    for step in range(steps):
        sd = stddev((steps - step) / steps)
        x = (torch.randn(batch_size, d) * sd).to(device)
        x.requires_grad_(True)
        t = T * torch.rand(batch_size).to(device)
        t.requires_grad_(True)
        
        optimizer.zero_grad()
        initial_loss = pde.initial_loss(model, x, device)
        dynamic_loss = pde.dynamic_loss(model, t, x, device)
        loss = initial_loss + dynamic_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (step + 1) % 100 == 0:
            print(f"Epoch [{step+1}/{steps}], Total Loss: {loss.item():.4f}, "
                  f"PDE Loss: {dynamic_loss.item():.4f}, "
                  f"Initial Loss: {initial_loss.item():.4f}")
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model

# Training function
# def train(φ, ρ, f, model, T, d, steps=1000, batch_size=100, lr=0.001, stddev=lambda t: 1.0, lr_decay=0.999,save_path=None):
#     model.apply(init_weights)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
#     dev = 'cpu'
    
#     for step in range(steps):
#         # Generate collocation points
#         sd = stddev((steps - step) / steps)
#         x = (torch.randn(batch_size, d) * sd).to(dev)
#         x.requires_grad_(True)
#         t = T * torch.rand(batch_size).to(dev)
#         t.requires_grad_(True)
        
#         optimizer.zero_grad()
#         # Compute losses
#         initial_loss = dgm_initial_loss(model, φ, x, dev)
#         dynamic_loss = dgm_dynamic_loss(model, ρ, f, t, x, dev)
#         loss = initial_loss + dynamic_loss
        
#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         if (step + 1) % 100 == 0:
#             print(f"Epoch [{step+1}/{steps}], Total Loss: {loss.item():.4f}, "
#                   f"PDE Loss: {dynamic_loss.item():.4f}, "
#                   f"Initial Loss: {initial_loss.item():.4f}")
    
#     if save_path:
#         torch.save(model.state_dict(), save_path)
#         print(f"Model saved to {save_path}")
        
#     return model


def calculate_energy_efficient(d,n,input_dim=1,output_dim=1):
    '''n is hidden_node_units
    N is len of networks'''

    input_macs = input_dim*n 

    hidden_macs = (d-1)*n*n 

    output_macs = n*output_dim

    total_energy = (input_macs+hidden_macs+output_macs)*4.6
    
    return total_energy


# PDE initial conditions
def φ1(x):
    return (1. + x.square().sum(dim=1, keepdim=True)).sqrt()

def φ2(x):
    return 2. / (4. + x.square().sum(dim=1, keepdim=True))

def φ3(x):
    return torch.atan(x.square().sum(dim=1, keepdim=True).sqrt() / 2.)

# PDE nonlinearity
def f(y):
    return torch.sin(y)

# Main execution
def main():
    from collections import defaultdict
    config_file = '/Users/user/Desktop/DeepPDE/paper3_dgm/config.cfg'
    base_output_path = '/Users/user/Desktop/DeepPDE/paper3_dgm/output_results'
    base_model_path = '/Users/user/Desktop/DeepPDE/paper3_dgm/model_output'
    T = 0.5
    ρ = 1
    num_epochs = 1000
    batch_size = 256
    true_value = 1.0  # Replace with actual true value for u(T, 0, ..., 0)
    modes = ['train','deploy']
    # modes = ['train']
    # modes = ['deploy']


    runs = 2
    initial_functions = {'phi1': φ1, 'phi2': φ2, 'phi3': φ3}

    pde_definitions = {
        'sinegordon1': SineGordon1(ρ=1.0, nonlinearity=lambda y: torch.sin(y))
    }


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ann_module = importlib.import_module('paper2_ThreeWays.ann_model')
    # DeepGalerkinNet = getattr(ann_module, 'DeepGalerkinNet')
    
    combinations = load_config_combinations(config_file)
    
    for mode in modes:
        for pde_name, pde in pde_definitions.items():

            for combo in combinations:
                model_name = combo['model']
                if model_name != 'ann':
                    continue

                module = importlib.import_module(f"models.{model_name}")
                DeepGalerkinNet = getattr(module, 'DeepGalerkinNet')

                
                d = combo['dimension']
                num_hidden_layers = combo['hidden_layers']
                num_hidden_units = combo['num_hidden_units']
                T_sim = combo['num_hidden_units']

                output_dir = os.path.join(base_output_path, pde_name, model_name)
                model_dir = os.path.join(base_model_path, pde_name, model_name)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)

                
                model = DeepGalerkinNet(d=d, num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units).to(device)
                
                if mode == 'train':
                    for run in range(runs):
                        # save_path = f'/Users/user/Desktop/DeepPDE/paper3_dgm/model_output/{model_name}_{combo["hidden_layers"]}_{combo["num_hidden_units"]}_{combo["T_sim"]}_d{combo["dimension"]}.pth'
                        save_path = os.path.join(model_dir, f"{run}_d{d}_{num_hidden_layers}_{num_hidden_units}_{T_sim}.pth")

                        # model = train(
                        #     φ=φ,
                        #     ρ=ρ,
                        #     f=f,
                        #     model=model,
                        #     T=T,
                        #     d=d,
                        #     steps=num_epochs,
                        #     batch_size=batch_size,
                        #     save_path = save_path
                        # )

                        model = train(
                            pde=pde,
                            model=model,
                            T=T,
                            d=d,
                            steps=num_epochs,
                            batch_size=batch_size,
                            save_path=save_path,
                            device=device
                        )
                elif mode == 'deploy':

                    energy = calculate_energy_efficient(d=num_hidden_layers,n=num_hidden_units,input_dim=d+1,output_dim=1)
                    predictions = []
                    absolute_error = []

                    result_dict = defaultdict(list)

                    df_true_value = pd.read_csv(f'/Users/user/Desktop/DeepPDE/paper3_dgm/{pde_name}_mlp.csv')

                    true_value = df_true_value[(df_true_value['d']==d)][' result_0'].iloc[0]


                    for run in range(runs):
                        model_path = os.path.join(model_dir, f"{run}_d{d}_{num_hidden_layers}_{num_hidden_units}_{T_sim}.pth")
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.eval()  # Set to evaluation mode for inference
                        with torch.no_grad():
                            pred = model(torch.tensor([[T] + [0.0] * d]).to(device)).item()
                            predictions.append(pred)
                            absolute_error.append(abs(pred-true_value))
                    
                    result_dict['mean'] = [np.mean(predictions)]
                    result_dict['std'] = [np.std(predictions)]
                    result_dict['energy'] = [energy]
                    result_dict['mean_abs_err'] = [np.mean(absolute_error)]


                    result_df = pd.DataFrame(result_dict)
                    result_df.to_csv(os.path.join(output_dir, f"results_d{d}_{num_hidden_layers}_{num_hidden_units}_{T_sim}.csv"))
                    # result_df.to_csv(f'/Users/user/Desktop/DeepPDE/paper3_dgm/output_results/{model_name}_{combo["hidden_layers"]}_{combo["num_hidden_units"]}_{combo["T_sim"]}_d{combo["dimension"]}.csv')


                # report_performance(model, φ1, T, d, device, true_value, os.path.join(output_path, f'dim_{d}'))

if __name__ == "__main__":
    main()