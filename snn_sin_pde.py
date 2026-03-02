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
        
        param_combinations = list(product(*params.values()))
        param_names = params.keys()
        
        for combo in param_combinations:
            config_combo = dict(zip(param_names, combo))
            config_combo['model'] = model
            combinations.append(config_combo)
    
    return combinations

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.normal_(m.bias, std=0.1)

def calculate_energy(acs, macs):
    E_AC = 0.9
    E_MAC = 4.6
    return E_AC * acs + E_MAC * macs

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

def report_performance(model, pde, T, d, device, true_value, output_folder):
    model.eval()
    with torch.no_grad():
        t = torch.tensor([[T]], dtype=torch.float32).to(device)
        x = torch.zeros(1, d).to(device)
        inputs = torch.cat((t, x), dim=1)
        predicted = model(inputs).item()
        
        abs_l1_error = abs(predicted - true_value)
        rel_l1_error = abs_l1_error / abs(true_value) if true_value != 0 else float('inf')
        
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

def main():
    from collections import defaultdict
    
    config_file = '/Users/user/Desktop/DeepPDE/paper3_dgm/config.cfg'
    base_output_path = '/Users/user/Desktop/DeepPDE/paper3_dgm/output_results'
    base_model_path = '/Users/user/Desktop/DeepPDE/paper3_dgm/model_output'
    T = 0.5
    num_epochs = 1000
    batch_size = 256
    # true_value = 1.0
    modes = ['train','deploy']
    # modes = ['deploy']

    runs = 2
    
    # pde_definitions = {
    #     'heat_pde': HeatPDE(ρ=1.0, nonlinearity=lambda y: torch.sin(y)),
    #     'alt_pde': AlternativePDE(ρ=0.5, nonlinearity=lambda y: torch.cos(y))
    # }

    pde_definitions = {
        'sinegordon1': SineGordon1(ρ=1.0, nonlinearity=lambda y: torch.sin(y))
    }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Check if GPU is activated
    if device.type == "mps":
        x = torch.rand(3, 3).to(device)
        print(f"GPU is active. Example tensor on {device}: {x}")
    else:
        print("GPU is not available, using CPU.")

    
    
    combinations = load_config_combinations(config_file)
    
    for mode in modes:
        for pde_name, pde in pde_definitions.items():
            for combo in combinations:
                model_name = combo['model']
                if model_name == 'ann':
                    continue
                
                module = importlib.import_module(f"models.{model_name}")
                DeepGalerkinNet = getattr(module, 'SpikingDeepGalerkinNet')
                
                d = combo['dimension']
                num_hidden_layers = combo['hidden_layers']
                num_hidden_units = combo['num_hidden_units']
                T_sim = combo['T_sim']
                
                output_dir = os.path.join(base_output_path, pde_name, model_name)
                model_dir = os.path.join(base_model_path, pde_name, model_name)
                os.makedirs(output_dir, exist_ok=True)
                os.makedirs(model_dir, exist_ok=True)
                
                
                if mode == 'train':
                    for run in range(runs):
                        save_path = os.path.join(model_dir, f"{run}_d{d}_{num_hidden_layers}_{num_hidden_units}_{T_sim}.pth")
                        # if os.path.exists(save_path):
                        #     print(f"Path exists: {save_path}")
                        #     continue

                        model = DeepGalerkinNet(d=d, num_hidden_layers=num_hidden_layers, num_hidden_units=num_hidden_units).to(device)

                  
                        
                        model_train = train(
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
                    predictions = []
                    energies = []
                    result_dict = defaultdict(list)
                    absolute_error = []

                    df_true_value = pd.read_csv(f'/Users/user/Desktop/DeepPDE/paper3_dgm/{pde_name}_mlp.csv')
                    true_value = df_true_value[(df_true_value['d']==d)][' result_0'].iloc[0]

                    for run in range(runs):
                        model_path = os.path.join(model_dir, f"{run}_d{d}_{num_hidden_layers}_{num_hidden_units}_{T_sim}.pth")

                        # Create a fresh model instance with the correct parameters from config
                        model = DeepGalerkinNet(
                            d=d,
                            num_hidden_layers=num_hidden_layers,
                            num_hidden_units=num_hidden_units,
                            T_sim=T_sim,  # Pass T_sim from config!
                            device=device
                        ).to(device)

                        # Load the trained weights
                        model.load_state_dict(torch.load(model_path, map_location=device))
                        model.eval()

                        with torch.no_grad():
                            pred = model(torch.tensor([[T] + [0.0] * d]).to(device)).item()
                            acs, macs = model.calculate_acs_macs_ops()
                            energy_consumption = calculate_energy(acs, macs)
                            absolute_error.append(abs(pred-true_value))
                            predictions.append(pred)
                            energies.append(energy_consumption)

                    result_dict['mean'] = [np.mean(predictions)]
                    result_dict['std'] = [np.std(predictions)]
                    result_dict['energy'] = [np.mean(energies)]
                    result_dict['mean_abs_err'] = [np.mean(absolute_error)]

                    result_df = pd.DataFrame(result_dict)
                    result_df.to_csv(os.path.join(output_dir, f"results_d{d}_{num_hidden_layers}_{num_hidden_units}_{T_sim}.csv"))
                # report_performance(model, pde, T, d, device, true_value, output_dir)

main()
# if __name__ == "__main__":
#     main()