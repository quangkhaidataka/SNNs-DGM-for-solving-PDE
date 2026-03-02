from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import numpy as np
import configparser
import importlib
import os
from itertools import product
import pandas as pd

# Abstract base class for PDE definitions
class PDE(ABC):
    @abstractmethod
    def initial_condition(self, x):
        """Define the initial condition φ(x)."""
        pass

    @abstractmethod
    def initial_loss(self, model, x, device):
        """Compute the initial loss: 𝔼[|v(0,x) - φ(x)|²]."""
        pass

    @abstractmethod
    def dynamic_loss(self, model, t, x, device):
        """Compute the dynamic loss based on the PDE."""
        pass

# Concrete PDE implementation for the given PDE
class SineGordon1(PDE):
    def __init__(self, ρ=1.0, nonlinearity=lambda y: torch.sin(y)):
        self.ρ = ρ
        self.f = nonlinearity

    def initial_condition(self, x):
        """Initial condition: φ(x) = √(1 + ||x||²)"""
        return (1. + x.square().sum(dim=1, keepdim=True)).sqrt()


    def initial_loss(self, model, x, device):
        """Compute initial loss: 𝔼[|v(0,x) - φ(x)|²]"""
        t = torch.zeros(x.shape[0], 1).to(device)
        u = model(torch.cat((t, x), dim=1))
        return (u - self.initial_condition(x)).square().mean()

    def dynamic_loss(self, model, t, x, device):
        """Compute dynamic loss: 𝔼[|∂v/∂t(t,x) - ρΔₓv(t,x) - f(v(t,x))|²]"""
        d = x.shape[1]
        x_comp = [x[:, i] for i in range(d)]
        u = model(torch.cat([t[:, None]] + [x_comp[i][:, None] for i in range(d)], dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        Δu = torch.zeros(x.shape[0]).to(device)
        for i in range(d):
            u_xi = torch.autograd.grad(u, x_comp[i], torch.ones_like(u), create_graph=True)[0]
            u_xixi = torch.autograd.grad(u_xi, x_comp[i], torch.ones_like(u_xi), create_graph=True)[0]
            Δu += u_xixi
        return (u_t - self.ρ * Δu - self.f(u)).square().mean()
    

class SemilinearHeatPDE(PDE):
    def __init__(self):
        self.ρ = 1.0  # Default diffusion coefficient

    def initial_condition(self, x):
        """Initial condition: u(x, t=0) = 5/(10 + 2||x||²)"""
        return 5.0 / (10.0 + 2.0 * x.square().sum(dim=1, keepdim=True))

    def initial_loss(self, model, x, device):
        """Compute initial loss: 𝔼[|v(0,x) - φ(x)|²]"""
        t = torch.zeros(x.shape[0], 1).to(device)
        u = model(torch.cat((t, x), dim=1))
        return (u - self.initial_condition(x)).square().mean()

    def dynamic_loss(self, model, t, x, device):
        """Compute dynamic loss: 𝔼[|∂v/∂t(t,x) - Δu(x,t) - (1 - u(t,x)²)/(1 + u(t,x)²)|²]"""
        d = x.shape[1]
        x_comp = [x[:, i] for i in range(d)]
        u = model(torch.cat([t[:, None]] + [x_comp[i][:, None] for i in range(d)], dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        Δu = torch.zeros(x.shape[0]).to(device)
        for i in range(d):
            u_xi = torch.autograd.grad(u, x_comp[i], torch.ones_like(u), create_graph=True)[0]
            u_xixi = torch.autograd.grad(u_xi, x_comp[i], torch.ones_like(u_xi), create_graph=True)[0]
            Δu += u_xixi
        nonlinearity = (1.0 - u.square()) / (1.0 + u.square())
        return (u_t - Δu - nonlinearity).square().mean()
    

class AllenCahnPDE(PDE):
    def __init__(self):
        self.ρ = 1.0  # Default diffusion coefficient

    def initial_condition(self, x):
        """Initial condition: u(x, t=0) = arctan(max(x_i, x_i))"""
        return torch.atan(x.abs().max(dim=1, keepdim=True)[0])

    def initial_loss(self, model, x, device):
        """Compute initial loss: 𝔼[|v(0,x) - φ(x)|²]"""
        t = torch.zeros(x.shape[0], 1).to(device)
        u = model(torch.cat((t, x), dim=1))
        return (u - self.initial_condition(x)).square().mean()

    def dynamic_loss(self, model, t, x, device):
        """Compute dynamic loss: 𝔼[|∂v/∂t(t,x) - Δu(x,t) - u(t,x) + u(t,x)³|²]"""
        d = x.shape[1]
        x_comp = [x[:, i] for i in range(d)]
        u = model(torch.cat([t[:, None]] + [x_comp[i][:, None] for i in range(d)], dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        Δu = torch.zeros(x.shape[0]).to(device)
        for i in range(d):
            u_xi = torch.autograd.grad(u, x_comp[i], torch.ones_like(u), create_graph=True)[0]
            u_xixi = torch.autograd.grad(u_xi, x_comp[i], torch.ones_like(u_xi), create_graph=True)[0]
            Δu += u_xixi
        nonlinearity = u - u.pow(3)
        return (u_t - Δu - nonlinearity).square().mean()


# Additional PDE example
class AlternativePDE(PDE):
    def __init__(self, ρ=0.5, nonlinearity=lambda y: torch.cos(y)):
        self.ρ = ρ
        self.f = nonlinearity

    def initial_condition(self, x):
        """Initial condition: φ(x) = 2 / (4 + ||x||²)"""
        return (1. + x.square().sum(dim=1, keepdim=True)).sqrt()


    def initial_loss(self, model, x, device):
        t = torch.zeros(x.shape[0], 1).to(device)
        u = model(torch.cat((t, x), dim=1))
        return (u - self.initial_condition(x)).square().mean()

    def dynamic_loss(self, model, t, x, device):
        d = x.shape[1]
        x_comp = [x[:, i] for i in range(d)]
        u = model(torch.cat([t[:, None]] + [x_comp[i][:, None] for i in range(d)], dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        Δu = torch.zeros(x.shape[0]).to(device)
        for i in range(d):
            u_xi = torch.autograd.grad(u, x_comp[i], torch.ones_like(u), create_graph=True)[0]
            u_xixi = torch.autograd.grad(u_xi, x_comp[i], torch.ones_like(u_xi), create_graph=True)[0]
            Δu += u_xixi
        return (u_t - self.ρ * Δu - self.f(u)).square().mean()
