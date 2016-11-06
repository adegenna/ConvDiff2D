import numpy as np
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt

# ***************************
# 2D CONVECTION DIFFUSION PDE SOLVER
# ***************************

class ConvDiff2D:
    # Constructor
    def __init__(self,NX,NY,NT):
        self.NX = NX;
        self.NY = NY;
        self.NT = NT;
        self.u  = 1.0;
        self.v  = 0.5;
        self.mu = 1.0e-3;
        self.grid();

    # Mesh initialization
    def grid(self):
        self.U     = np.zeros(self.NX*self.NY);
        Xtmp       = np.linspace(0,1,self.NX);
        Ytmp       = np.linspace(0,1,self.NY);
        XX,YY      = np.meshgrid(Xtmp,Ytmp);
        self.X     = np.reshape(XX,self.NX*self.NY);
        self.Y     = np.reshape(YY,self.NX*self.NY);
        sig        = 0.2;
        # self.U     = np.exp(-0.5*np.sqrt((self.X-0.25)**2+(self.Y-0.25)**2)/(sig**2)) \
        #              + np.exp(-0.5*np.sqrt((self.X-0.75)**2+(self.Y-0.75)**2)/(sig**2))
        self.dx    = 1.0/(self.NX-1);
        self.dy    = 1.0/(self.NY-1);
        self.dt    = 3.0/(self.NT-1);
        self.T     = np.linspace(0,3.0,self.NT);

    # Gaussian process for injection profile
    def initializeInjectors(self):
        MEAN          = 2.0;
        COV           = np.zeros([self.NT,self.NT]);
        L             = 0.1;
        self.numInj   = 5;
        self.centersX = np.array([0.2, 0.2, 0.2, 0.2, 0.2]);
        self.centersY = np.array([0.1, 0.3, 0.5, 0.7, 0.9]);
        for i in range(0,self.NT):
            COV[i] = (1.0**2)*np.exp(-0.5*((self.T[i]-self.T)**2)/(L**2));
        lam,phi = np.linalg.eig(COV);
        self.Ji    = MEAN + np.zeros([self.numInj,self.NT])
        for i in range(0,self.numInj):
            for j in range(0,self.NT):
                self.Ji[i] += np.sqrt(lam[j])*phi[:,j]*np.random.normal();

    # Generate injection profile
    def generateInjection(self,timestep):
        sigX = 0.05;
        sigY = 0.05;
        J    = np.zeros([self.NX*self.NY]);
        for i in range(0,self.numInj):
            JX = np.exp(-0.5*((self.X-self.centersX[i])**2)/(sigX**2));
            JY = np.exp(-0.5*((self.Y-self.centersY[i])**2)/(sigY**2));
            J += self.Ji[i,timestep]*JX*JY;
        return J;
            
    # ADI routine
    def ADI(self,timestep):
        # First half step (derivatives in x)
        D2row   = np.diag(-2*np.ones(self.NX)) \
                 + np.diag(1*np.ones(self.NX-1),1) \
                 + np.diag(1*np.ones(self.NX-1),-1);
        D2row   = self.mu/(self.dx**2)*D2row;
        D2X     = np.zeros([self.NX*self.NY , self.NX*self.NY]);
        D1row   = np.identity(self.NX) \
                  + np.diag(-1*np.ones(self.NX-1),-1);
        D1row   = self.u/self.dx*D1row;
        D1X     = np.zeros([self.NX*self.NY , self.NX*self.NY]);
        for i in range(0,self.NY):
            D2X[i*self.NX:(i+1)*self.NX , i*self.NX:(i+1)*self.NX] = D2row;
            D1X[i*self.NX:(i+1)*self.NX , i*self.NX:(i+1)*self.NX] = D1row;
        D2col  = np.diag(-2*np.ones(self.NY)) \
                 + np.diag(1*np.ones(self.NY-1),1) \
                 + np.diag(1*np.ones(self.NY-1),-1);
        D2col  = self.mu/(self.dy**2)*D2col;
        D2Y    = np.zeros([self.NX*self.NY , self.NX*self.NY]);
        D1col  = np.identity(self.NY) \
                 + np.diag(-1*np.ones(self.NY-1),-1);
        D1col  = self.v/self.dy*D1col;
        D1Y    = np.zeros([self.NX*self.NY , self.NX*self.NY]);
        for i in range(0,self.NX):
            D2Y[i*self.NY:(i+1)*self.NY , i*self.NY:(i+1)*self.NY] = D2col;
            D1Y[i*self.NY:(i+1)*self.NY , i*self.NY:(i+1)*self.NY] = D1col;
        LHS = (2./self.dt)*np.identity(self.NX*self.NY) - D2X + D1X;
        Uprev_Ymajor = np.zeros(self.NX*self.NY);
        for j in range(0,self.NX):
            Uprev_Ymajor[j*self.NY:(j+1)*self.NY] = \
                self.U[j:self.NX*self.NY:self.NX];
        D2YU = np.dot(D2Y,Uprev_Ymajor);
        D2YU_xmajor = np.zeros(self.NX*self.NY);
        for j in range(0,self.NY):
            D2YU_xmajor[j*self.NX:(j+1)*self.NX] = \
                D2YU[j:self.NX*self.NY:self.NY];
        J_xmajor = self.generateInjection(timestep);
        RHS      = (2./self.dt)*self.U + D2YU_xmajor + 0.5*J_xmajor;
        U_half = np.linalg.solve(LHS,RHS);
        # Second half step (derivatives in y)
        LHS = (2./self.dt)*np.identity(self.NX*self.NY) - D2Y + D1Y;
        Uprev_Ymajor = np.zeros(self.NX*self.NY);
        for j in range(0,self.NX):
            Uprev_Ymajor[j*self.NY:(j+1)*self.NY] = \
                U_half[j:self.NX*self.NY:self.NX];
        D2XU = np.dot(D2X,U_half);
        D2XU_ymajor = np.zeros(self.NX*self.NY);
        J_ymajor    = np.zeros(self.NX*self.NY);
        for j in range(0,self.NX):
            D2XU_ymajor[j*self.NY:(j+1)*self.NY] = \
                D2XU[j:self.NX*self.NY:self.NX];
            J_ymajor[j*self.NY:(j+1)*self.NY] = \
                J_xmajor[j:self.NX*self.NY:self.NX];
        RHS = (2./self.dt)*Uprev_Ymajor + D2XU_ymajor + 0.5*J_ymajor;
        U_new = np.linalg.solve(LHS,RHS);
        U_new_xmajor = np.zeros(self.NX*self.NY);
        for j in range(0,self.NY):
            U_new_xmajor[j*self.NX:(j+1)*self.NX] = \
                U_new[j:self.NX*self.NY:self.NY];
        self.U = U_new_xmajor.copy();
        
    # Main solve routine
    def solve(self):
        self.initializeInjectors();
        for i in range(0,self.NT):
            print i
            self.ADI(i);
            filename = "SOLN" + str(i) + '.out';
            np.save(filename,self.U);
