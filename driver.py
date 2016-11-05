import numpy as np
import matplotlib.pyplot as plt
from ConvDiff2D import ConvDiff2D

NX = 20;
NY = 20;
NT = 500;
pde = ConvDiff2D(NX,NY,NT)
pde.solve();
