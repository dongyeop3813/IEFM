from abc import ABC, abstractmethod

import numpy as np
import torch


class BaseNoiseSchedule(ABC):
    @abstractmethod
    def g(t):
        # Returns g(t)
        pass

    @abstractmethod
    def h(t):
        # Returns \int_0^t g(t)^2 dt
        pass


class LinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.full_like(t, self.beta**0.5)

    def h(self, t):
        return self.beta * t


class QuadraticNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.sqrt(self.beta * 2 * t)

    def h(self, t):
        return self.beta * t**2


class PowerNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta, power):
        self.beta = beta
        self.power = power

    def g(self, t):
        return torch.sqrt(self.beta * self.power * (t ** (self.power - 1)))

    def h(self, t):
        return self.beta * (t**self.power)


class SubLinearNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, beta):
        self.beta = beta

    def g(self, t):
        return torch.sqrt(self.beta * 0.5 * 1 / (t**0.5 + 1e-3))

    def h(self, t):
        return self.beta * t**0.5


class GeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = self.sigma_max / self.sigma_min

    def g(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then g(t) = sigma_min * sigma_d^t * sqrt{2 * log(sigma_d)}
        # See Eq 192 in https://arxiv.org/pdf/2206.00364.pdf
        return (
            self.sigma_min
            * (self.sigma_diff**t)
            * ((2 * np.log(self.sigma_diff)) ** 0.5)
        )

    def h(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_min * (((self.sigma_diff ** (2 * t)) - 1) ** 0.5)) ** 2


class OTcnfNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max - sigma_min

    def g(self, t):
        raise NotImplementedError

    def h(self, t):
        # Noise schedule of OT probability path
        # see Eq 20 in (Flow matching, Lipman et al., 2023)
        return (self.sigma_max - self.sigma_diff * t) ** 2

    def sigma(self, t):
        return self.sigma_max - self.sigma_diff * t

    def sigma_prime(self, t):
        return -torch.full_like(t, self.sigma_diff)


class VEcnfNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_min / sigma_max
        self.c = np.log(self.sigma_diff)

    def g(self, t):
        raise NotImplementedError

    def h(self, t):
        # Noise schedule of OT probability path
        # see Eq 20 in (Flow matching, Lipman et al., 2023)
        return (self.sigma_max * ((self.sigma_diff) ** t)) ** 2

    def sigma(self, t):
        return self.sigma_max * ((self.sigma_diff) ** t)

    def sigma_prime(self, t):
        return self.sigma_max * self.c * ((self.sigma_diff) ** t)


class RevGeometricNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max / sigma_min

    def g(self, t):
        raise NotImplementedError

    def h(self, t):
        # Let sigma_d = sigma_max / sigma_min
        # Then h(t) = \int_0^t g(z)^2 dz = sigma_min * sqrt{sigma_d^{2t} - 1}
        # see Eq 199 in https://arxiv.org/pdf/2206.00364.pdf
        return (self.sigma_max * ((self.sigma_diff) ** (1 - t))) ** 2


class SqrtNoiseSchedule(BaseNoiseSchedule):
    def __init__(self, sigma_min, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_diff = sigma_max**2 - sigma_min**2

    def g(self, t):
        raise NotImplementedError

    def h(self, t):
        return self.sigma_max**2 - self.sigma_diff * t

    def sigma(self, t):
        return (self.sigma_max**2 - self.sigma_diff * t) ** 0.5

    def sigma_prime(self, t):
        return -self.sigma_diff / (2 * self.sigma(t))
