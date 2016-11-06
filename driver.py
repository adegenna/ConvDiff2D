import numpy as np
import matplotlib.pyplot as plt
from ConvDiff2D import ConvDiff2D

NX = 40;
NY = 40;
NT = 500;
pde = ConvDiff2D(NX,NY,NT)
pde.solve();
