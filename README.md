# ⚡ SNN-DGM: Spiking Neural Networks for Energy-Efficient PDE Solving

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![C++](https://img.shields.io/badge/C++-Energy%20Profiler-00599C?style=flat-square&logo=c%2B%2B)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![Domain](https://img.shields.io/badge/Domain-Scientific%20Computing-teal?style=flat-square)
![Approach](https://img.shields.io/badge/Approach-Neuromorphic%20Computing-green?style=flat-square)

---

## 📌 Overview

Solving high-dimensional Partial Differential Equations (PDEs) is a fundamental challenge across physics, engineering, and quantitative finance. Classical numerical methods such as finite difference and finite element methods suffer from the **curse of dimensionality** — their computational cost grows exponentially with the number of dimensions, making them intractable for real-world high-dimensional problems.

The **Deep Galerkin Method (DGM)**, introduced by Sirignano & Spiliopoulos (2018), addresses this by using neural networks as universal function approximators to learn PDE solutions directly — replacing the mesh-based computation with stochastic gradient descent over randomly sampled interior and boundary points.

However, conventional DGM relies on **Artificial Neural Networks (ANNs)**, which are computationally dense and energy-expensive. This project proposes replacing the ANN in DGM with a **Spiking Neural Network (SNN)** — a biologically-inspired, event-driven architecture that operates via sparse binary spikes rather than continuous activations.

The key results demonstrate that **SNN-DGM**:
- Achieves **comparable solution accuracy** to ANN-DGM across tested PDEs
- Delivers approximately **3× better energy efficiency**, measured via theoretical synaptic operations (ACs vs MACs)

---

## 🎯 Objectives

- Implement the Deep Galerkin Method (DGM) for solving PDEs using both ANN and SNN architectures
- Apply both approaches to the **Sine-Gordon PDE** as a benchmark problem
- Compare solution accuracy (mean forecast error, standard deviation) between SNN-DGM and ANN-DGM
- Measure theoretical energy consumption by counting Accumulate operations (ACs) for SNNs vs. Multiply-Accumulate operations (MACs) for ANNs
- Demonstrate that neuromorphic computing can match ANN accuracy at a fraction of the energy cost in scientific computing

---

## 💡 Key Concepts

### Deep Galerkin Method (DGM)
DGM treats a PDE solution as a neural network approximation trained to satisfy:
- **Initial conditions** (loss at t = 0)
- **Boundary conditions** (loss on the domain boundary)
- **PDE dynamics** (loss from the differential equation residual at randomly sampled interior points)

The network is trained by minimising the sum of these three loss components, effectively learning the solution function directly without a computational mesh.

### Why Spiking Neural Networks?
Unlike ANNs that perform dense floating-point multiplications at every layer, SNNs communicate through discrete binary spikes — firing only when a neuron's membrane potential crosses a threshold. This leads to drastically fewer arithmetic operations per inference pass:

| Property | ANN-DGM | SNN-DGM |
|---|---|---|
| Activation type | Continuous (dense float) | Binary spikes (sparse) |
| Core operation | MACs (Multiply-Accumulate) | ACs (Accumulate only) |
| Energy per operation | ~4–10× higher | Baseline |
| Solution accuracy | Baseline | ✅ Comparable |
| Energy efficiency | Baseline | ✅ ~3× lower |
| Neuron model | ReLU / Tanh | Leaky Integrate-and-Fire (LIF) |

### Energy Measurement Methodology
Energy efficiency is estimated theoretically by counting the number of synaptic operations:
- **ANN layers** → counted as **MACs** (Multiply-Accumulate operations, energy-expensive)
- **SNN layers** → counted as **ACs** (Accumulate-only operations, ~4× cheaper per op on neuromorphic hardware)

This provides a hardware-agnostic, principled comparison of computational cost between the two architectures.

---

## 🔬 Methodology

### Step 1 — PDE Definition (`pde_class.py`)
PDEs are defined as Python classes. Each class encapsulates:
- `initial_condition`: defines the initial state of the system
- `cal_dynamic_loss`: computes the PDE residual loss at interior points
- `cal_initial_loss`: computes the loss at the initial time step

Currently implemented PDEs include the **Sine-Gordon equation**, with the class-based design making it straightforward to extend to other PDEs (e.g., Black-Scholes, Hamilton-Jacobi-Bellman).

### Step 2 — Model Architecture (`models/`)
The `models/` folder contains SNN class definitions. The core model is based on:
- **Leaky Integrate-and-Fire (LIF)** neurons as the spiking activation mechanism
- Architecture inspired by the paper *"Spiking Neural Networks for Nonlinear Regression"* (Henkes et al.)
- A custom `calculate_acs_macs_ops` function that counts ACs and MACs per layer to estimate theoretical energy

### Step 3 — Training & Evaluation
Two parallel scripts run the full DGM pipeline, one for each architecture:

| Script | Architecture | Description |
|---|---|---|
| `snn_sin_pde.py` | SNN | DGM with SNN as the prediction model |
| `ann_sin_pde.py` | ANN | DGM with ANN as the prediction model (baseline) |

Both scripts support two execution modes:
- **`train` mode**: Trains the model for 20 independent runs, saving a model object per run to `model_output/`
- **`deploy` mode**: Loads saved models, generates predictions, and computes theoretical energy consumption

**Output** (deploy mode): A CSV file containing mean forecast, standard deviation, and theoretical energy per architecture for direct comparison.

### Step 4 — Interactive Demo (`demo.ipynb`)
A Jupyter notebook providing a visual, step-by-step walkthrough of the full experiment — from PDE setup and training to solution plots and energy comparison charts.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+, C++ |
| Deep Learning | `PyTorch` |
| SNN Framework | `snnTorch` |
| Numerical Computing | `numpy`, `scipy` |
| Visualisation | `matplotlib` |
| Configuration | `config.cfg` (INI format) |
| Energy Profiling | `mlp_v2_onehalf.cpp` (custom C++ MAC/AC counter) |
| Demo | Jupyter Notebook (`demo.ipynb`) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A C++ compiler (e.g., `g++`) for building the energy profiling utility
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/quangkhaidataka/SNNs-DGM-for-solving-PDE.git
cd SNNs-DGM-for-solving-PDE
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch snnTorch numpy scipy matplotlib configparser jupyter
```

### 4. Configure Experiment Parameters

Edit `config.cfg` to set hyperparameters before running:

```ini
[parameters]
no_of_layer = 4          # Number of hidden layers
node_of_layer = 50       # Neurons per layer
T = 1.0                  # Time horizon for PDE
```

---

## ▶️ Running Experiments

### Train the SNN-DGM Model

```bash
python snn_sin_pde.py --mode train
```

Trains 20 independent runs. Saved model objects are written to `model_output/`.

### Train the ANN-DGM Baseline

```bash
python ann_sin_pde.py --mode train
```

### Deploy and Evaluate (Generate Results CSV)

```bash
python snn_sin_pde.py --mode deploy
python ann_sin_pde.py --mode deploy
```

Each command outputs a CSV file with mean forecast error, standard deviation, and theoretical energy consumption — ready for direct comparison between architectures.

### Run the Interactive Demo

```bash
jupyter notebook demo.ipynb
```

Provides a visual walkthrough of both models including solution surface plots, error analysis, and energy comparison charts.

---

## 📁 Project Structure

```
SNNs-DGM-for-solving-PDE/
│
├── snn_sin_pde.py               # DGM pipeline with SNN (Sine-Gordon PDE)
├── ann_sin_pde.py               # DGM pipeline with ANN (Sine-Gordon PDE) — baseline
├── pde_class.py                 # PDE class definitions (initial/boundary/dynamic loss)
├── config.cfg                   # Hyperparameter configuration file
├── demo.ipynb                   # Interactive demo notebook
├── mlp_v2_onehalf.cpp           # C++ utility for MAC/AC operation counting
│
├── models/                      # SNN model class definitions
│   └── snn_regression_torch_memberance.py   # Core SNN model with LIF neurons
│                                              # Includes calculate_acs_macs_ops()
│
└── README.md                    # Project documentation
```

---

## 📊 Results Summary

| Metric | ANN-DGM | SNN-DGM |
|---|---|---|
| Solution Accuracy | Baseline | ✅ Comparable |
| Energy Consumption | Baseline (MACs) | ✅ ~3× Lower (ACs) |
| Core Neuron Model | ReLU / Tanh | Leaky Integrate-and-Fire (LIF) |
| Operation Type | Multiply-Accumulate (MAC) | Accumulate only (AC) |

The SNN-DGM framework demonstrates that neuromorphic computing is a viable and energy-efficient alternative to conventional deep learning for solving complex scientific PDEs — opening a promising direction for sustainable AI in high-performance computing.

---

## 📚 References

This project builds upon and extends the following work:

> Sirignano, J., & Spiliopoulos, K. (2018). *DGM: A deep learning algorithm for solving partial differential equations.* Journal of Computational Physics.

For the SNN architecture and energy measurement methodology:

> Henkes, A., Eshraghian, J. K., & Wessels, H. (2022). *Spiking Neural Networks for Nonlinear Regression.* arXiv preprint.
> 🔗 [Reference implementation](https://github.com/ahenkes1/HENKES_SNN)

For background on Spiking Neural Networks:

> Maass, W. (1997). *Networks of Spiking Neurons: The Third Generation of Neural Network Models.* Neural Networks.

---

## 🤝 Contributing

Contributions and discussions are welcome. Feel free to open an issue for questions about the SNN-DGM architecture, PDE extensions, or energy measurement methodology. Pull requests for adding new PDE classes or SNN variants are especially encouraged.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Quang Khai**
- GitHub: [@quangkhaidataka](https://github.com/quangkhaidataka)
