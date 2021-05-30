'''
Written By: Abhijeet Kulkarni (amkulk@udel.edu)
For: Research Project MEEG678
Title: Autonomous driving using Model Predictive ContouringControl.
'''
import numpy as np

# importing saved dynamics functions
import dill
dill.settings['recurse'] = True
fode = dill.load(open("functions/eta_dot.dill","rb"))
# fF_lateral=dill.load(open("F_contact.dill", "rb"))

def fF_lateral(states,inputs,a1,a2,C1,C2):
    X,Y,theta,dX,dY,dtheta=states
    delta,Fx1,Fx2=inputs
    vx = (dX*np.cos(theta)+dY*np.sin(theta))
    vy = (-dX*np.sin(theta)+dY*np.cos(theta))
    if np.abs(vx)>0:
        if vx>0:
            Fy1=-C1*(-delta+np.arctan((vy+dtheta*a1)/vx))
            Fy2=-C2*(np.arctan((vy-dtheta*a2)/vx))
        else:
            Fy1=C1*(-delta+np.arctan((vy+dtheta*a1)/vx))
            Fy2=C2*(np.arctan((vy-dtheta*a2)/vx))

    else:
        Fy1 = C1*delta
        Fy2 = 0
    return np.array([Fy1,Fy2])

def warp90(angle):#angle=[-pi,pi]
    # warps angle to [-pi/2,pi/2]
    if (angle >= np.pi/2) or (angle <= -np.pi/2):
        return (angle%(np.pi/2))*np.sign(angle)
    else:
        return angle

class CarParameters():
    def __init__(self):
        # values from https://github.com/alexliniger/MPCC/blob/master/Matlab/getModelParams.m
        self.g = 9.81
        self.M  = 1573
        self.I  = 2873
        self.a1 = 0.2
        self.a2 = 1.35
        self.d1 = 10        #damping
        self.d2 = 100
        self.mu = 0.9
        self.D1 = self.mu * self.M * self.g * (self.a2/(self.a1 + self.a2))
        self.D2 = self.mu * self.M * self.g * (self.a1/(self.a1 + self.a2))
        self.C1 = 2
        self.C2 = 2
        self.B1 = 13
        self.B2 = 13
        self.Fxmax = 5000 #maximum force from one wheel
        self.deltamax = np.deg2rad(20)

class CarParametersSmall():
    def __init__(self):
        # values from https://github.com/alexliniger/MPCC/blob/master/Matlab/getModelParams.m
        self.g = 9.81
        self.M  = 0.041
        self.I  = 27.8e-6
        self.a1 = 0.029
        self.a2 = 0.033
        self.d1 = 0        #damping
        self.d2 = 0
        self.mu = 0.9
        self.D1 = self.mu * self.M * self.g * (self.a2/(self.a1 + self.a2))
        self.D2 = self.mu * self.M * self.g * (self.a1/(self.a1 + self.a2))
        self.C1 = 1.2
        self.C2 = 1.2691
        self.B1 = 2.579
        self.B2 = 3.3852
        self.Fxmax = 2 #maximum force from one wheel
        self.deltamax = np.deg2rad(50)

class CarDynamics(CarParametersSmall):
    def __init__(self,ini_states,dt):
        super().__init__()
        self.dt = dt
        self.ini_states = ini_states
        self.states = self.ini_states
        self.time = 0
  
    def reset(self):
        self.states = self.ini_states
    
    def F_LateralNew(self,inputs):#https://andresmendes.github.io/openvd/build/html/tire.html
        X,Y,theta,dX,dY,dtheta=self.states
        delta,Fx1,Fx2=inputs
        vx = ((dX*np.cos(theta)+dY*np.sin(theta)))
        vxmag = np.abs(vx)
        vxdir = np.sign(vx)
        vy = (-dX*np.sin(theta)+dY*np.cos(theta))
        if vxmag >0:
            alpha1 = vxdir*((np.arctan((vy+dtheta*self.a1)/vx))-delta)
            alpha2 = vxdir*(np.arctan((vy-dtheta*self.a2)/vx))
        else:
            alpha1 = -delta
            alpha2 = 0
        # print(warp90(np.arctan2((vy+dtheta*self.a1),vx)))
        Fy1 = -self.D1*np.sin(self.C1*np.arctan(self.B1*alpha1))
        Fy2 = -self.D2*np.sin(self.C2*np.arctan(self.B2*alpha2))
        return [Fy1,Fy2]

    def damping_forces(self,inputs):
        X,Y,theta,dX,dY,dtheta=self.states
        vx = (dX*np.cos(theta)+dY*np.sin(theta))
        Fx1 = -vx*self.d1+inputs[1]
        Fx2 = -vx*self.d2+inputs[2]
        return [inputs[0],Fx1,Fx2]

    def sim_step(self,inputs): #input = [delta,Fx1,Fx2]
        #Damping
        dampedinput = self.damping_forces(inputs)
        #contact force
        F_lateral = self.F_LateralNew(inputs)
        #deta
        #deta = fode(self.states,dampedinput,F_lateral,self.M,self.I,self.a1,self.a2).flatten()
        # ode4
        s1 = fode(self.states,dampedinput,F_lateral,self.M,self.I,self.a1,self.a2).flatten()
        s2 = fode(self.states+self.dt*s1/2,dampedinput,F_lateral,self.M,self.I,self.a1,self.a2).flatten()
        s3 = fode(self.states+self.dt*s2/2,dampedinput,F_lateral,self.M,self.I,self.a1,self.a2).flatten()
        s4 = fode(self.states+self.dt*s3,dampedinput,F_lateral,self.M,self.I,self.a1,self.a2).flatten()
        self.states = self.states + self.dt*(s1+2*s2+2*s3+s4)/6
        #self.states = [i*self.dt+s for i,s in zip(deta,self.states)]
        self.time +=self.dt


