import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

plt.xlim(-5, 5)
plt.ylim(-5, 5)

x0, y0 = 0, 0

# PDE solver stuff
NX = 40;
NY = 40;
NT = 500;
X = np.linspace(0,1,NX);
Y = np.linspace(0,1,NY);
XX,YY = np.meshgrid(X,Y);
levels = np.linspace(-0.025,0.34,100);
centersX = np.array([0.2, 0.2, 0.2, 0.2, 0.2]);
centersY = np.array([0.1, 0.3, 0.5, 0.7, 0.9]);

with writer.saving(fig, "ConvDiffSources2D.mp4", 100):
    for i in range(0,NT/10):
        print i
        plt.clf()
        filename = "SOLN" + str(10*i) + ".out.npy";
        SOLN = np.load(filename);
        plt.contourf(XX,YY,np.reshape(SOLN,[NX,NY]),levels=levels);
        plt.colorbar()
        plt.scatter(centersX,centersY,30,'k');
        plt.gca().set_xlim([0,1]);
        plt.gca().set_ylim([0,1]);
        #plt.ion(); plt.draw();
        #time.sleep(0.1);
        #l.set_data(x0, y0)
        writer.grab_frame()
