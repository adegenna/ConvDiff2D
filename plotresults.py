import numpy as np
import matplotlib.pyplot as plt
import time

NX = 40;
NY = 40;
NT = 500;
X = np.linspace(0,1,NX);
Y = np.linspace(0,1,NY);
XX,YY = np.meshgrid(X,Y);
levels = np.linspace(0,1.,60);
plt.figure(1);
for i in range(0,NT/10):
    plt.clf()
    filename = "SOLN" + str(10*i) + ".out.npy";
    SOLN = np.load(filename);
    plt.contourf(XX,YY,np.reshape(SOLN,[NX,NY]),levels=levels);
    plt.colorbar()
    plt.ion(); plt.draw();
    time.sleep(0.1);
