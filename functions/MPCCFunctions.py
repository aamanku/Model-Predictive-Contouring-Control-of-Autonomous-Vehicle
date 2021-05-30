'''
Written By: Abhijeet Kulkarni (amkulk@udel.edu)
For: Research Project MEEG678
Title: Autonomous driving using Model Predictive ContouringControl.
'''
import math
import numpy as np
from functions.HelpingFunctions import *
from functions.Car_Dynamics import *
# from HelpingFunctions import *
# from Car_Dynamics import *
import scipy as sp
import cvxpy as cp
import dill
import time
dill.settings['recurse'] = True


#####Getting functions generated in Dynamics.ipynb. 
# The generated symbolic functions are lambidified and save in file using dill package.

### Error Constraint
ErrorsLindata=dill.load(open("functions/ErrorsLindata.dill", "rb"))


### Discerete Linear Dynamics Constraint
DiscDynaMatrices = dill.load(open("functions/DiscDynaMatrices.dill", "rb"))


### alpha linearized
AlphaLindata = dill.load(open("functions/alpha2data.dill", "rb"))


###MPCC object


###Object for the MPCC controller

class MPCC_Object(CarParametersSmall):
    def __init__(self,N,Ts,xc,yc,xi,yi,xo,yo):
        super().__init__()  #get paramters
        self.N  = N          #number of time steps
        self.Ts = Ts        #total time
        self.dt = self.Ts/self.N    #time between steps
        self.alphaX = 0         #bezier coefficients
        self.alphaY = 0
        self.xc = xc #track
        self.yc = yc
        self.xi = xi
        self.yi = yi
        self.xo = xo
        self.yo = yo

    #dimensions
        self.N_states   = 6             #number of states fixed
        self.N_inputs   = 3             #number of inputs [delta,Fx1,Fx2]
        self.N_Lforce   = 2             #number of lateral forces [Fy1,Fy2]   
        self.N_errors   = 2             #number of errors [ec el]

    #weights
        self.We  = np.diag([5,100])
        self.Q_alpha  = np.array([[1]])
        self.Q_input = np.diag([1,10,0])
        self.umax = np.array([[self.deltamax],[0],[self.Fxmax]])
        self._cumax = np.array([[self.deltamax/20],[self.Fxmax/2],[self.Fxmax/2]])#constraint on change in u
        self._cqmax = np.array([[2],[2],[1],[50],[50],[10]])# constraint on change in q
        self.qmax_4_6 = np.array([[100],[100],[20]]) #limit on rates
        self.Rdu = np.diag([70,1,20])
        self.qv = np.array([[-30]])
        self.tamax = np.array([[1]])
        self.vmax = np.array([[(1)/self.dt]])
        self.qc_term = np.array([[0]])
        self.W_slack_si = 1000
        self.W_slack_so = 1000


    ##variables
        self.q  = self.MakeVarList(self.N+1,(self.N_states,1)) #states
        self.u  = self.MakeVarList(self.N,(self.N_inputs,1)) #inputs
        self.ta = self.MakeVarList(self.N+1,(1,1))            #approximate parameter of bezier curve
        self.v  = self.MakeVarList(self.N,(1,1))                #virtual progress factor input 
        ##initializing parameters

    #Dynamics 
        #linearized dynamics
        self.A_     = self.MakeParaList(self.N,(self.N_states,self.N_states)) #array of A_k
        self.B_     = self.MakeParaList(self.N,(self.N_states,self.N_inputs))
        self.G_     = self.MakeParaList(self.N,(self.N_states,1))
    #error
        #linearized error quadratic function
        self.Ae       = self.MakeParaList(self.N,(self.N_states+1,self.N_states+1),PSD = True)
        self.Ge       = self.MakeParaList(self.N,(1,self.N_states+1))
    #nominal
        #values around which lienarization is performed                                                   
        self._q     = self.MakeParaList(self.N+1,(self.N_states,1))#nominal states
        self._u     = self.MakeParaList(self.N,(self.N_inputs,1),value=np.array([[0.0],[0],[1]]))#nominal inputs
        self._ta    = self.MakeParaList(self.N+1,(1,1))#nominal bezier parameter
        self._v     = self.MakeParaList(self.N,(1,1),value=np.array([[1/self.dt/self.N-0.5]]))#nominal bezier parameter rate
    #boundary
        ##  # a_i*y+b_i*x+c_i>0
        self.a_i    = self.MakeParaList(self.N,(1,1))#inner
        self.b_i    = self.MakeParaList(self.N,(1,1))
        self.c_i    = self.MakeParaList(self.N,(1,1)) 
        self.a_o    = self.MakeParaList(self.N,(1,1))
        self.b_o    = self.MakeParaList(self.N,(1,1))#outer
        self.c_o    = self.MakeParaList(self.N,(1,1))
        self.s_i    = self.MakeVarList(self.N,(1,1))    #slack variable for soft constraint to recover from infeasibility
        self.s_o    = self.MakeVarList(self.N,(1,1))


    #slip angle of rear
        #alpha_=C+Gx linear approximation
        # A_alpha = 2CWG
        self.A_alpha = self.MakeParaList(self.N,(self.N_states,self.N_states))
        self.G_alpha = self.MakeParaList(self.N,(1,self.N_states))
    
    #Closeness to linearization
        self.Du = self.MakeVarList(self.N-1,(self.N_inputs,1)) #deviation from nominal input

        #############################################################################
        ##defining cost
        self.cost = 0
        for k in range(0,self.N):
            self.cost += cp.sum_squares(self.Ae[k]@cp.vstack([self.q[k+1],self.ta[k+1]]))+self.Ge[k]@cp.vstack([self.q[k+1],self.ta[k+1]]) + self.qv@self.v[k] + self.W_slack_si*cp.sum_squares(self.s_i[k]) + self.W_slack_so*cp.sum_squares(self.s_o[k])
            self.cost += cp.sum_squares(self.A_alpha[k]@self.q[k+1]) + self.G_alpha[k]@self.q[k+1]
            self.cost += cp.sum_squares(self.Q_input@self.u[k])
        for k in range(0,self.N-1):
            self.cost += cp.sum_squares(self.Rdu@self.Du[k])
        # self.cost += self.qc_term*cp.sum_squares(self.Ae[-1]@cp.vstack([self.q[-1],self.ta[-1]]))+self.qc_term*self.Ge[-1]@cp.vstack([self.q[-1],self.ta[-1]])
        #############################################################################

        ##defining constraints
        self.constraints = []
        self.constraints += [self.q[0]==self._q[0]]
        self.constraints += [self.ta[0]==self._ta[0]]
        for k in range(0,self.N):
            self.constraints += [self.q[k+1] == self.A_[k]@self.q[k] + self.B_[k]@self.u[k] + self.G_[k],
                                self.ta[k+1] == self.ta[k] + self.v[k]*self.dt,
                                self.a_i[k]@self.q[k+1][0]+self.b_i[k]@self.q[k+1][1]+self.c_i[k]>=self.s_i[k],
                                self.a_o[k]@self.q[k+1][0]+self.b_o[k]@self.q[k+1][1]+self.c_o[k]>=self.s_o[k]]
        
        for k in range(0,self.N):
            self.constraints += [0<=self.ta[k+1],self.ta[k+1]<=self.tamax,
                                 cp.abs(self._q[k+1]-self.q[k+1])<=self._cqmax,
                                 cp.abs(self._u[k]-self.u[k])<=self._cumax,
                                 cp.abs(self.u[k])<=self.umax,
                                 0<=self.v[k],self.v[k]<=self.vmax]
                                #  -self.qmax_4_6<=self.q[k+1][3:6],self.q[k+1][3:6]<=self.qmax_4_6]

        for k in range(self.N-1):
            self.constraints += [self.Du[k]==self.u[k+1]-self.u[k]]
        
        #############################################################################

        self.problem = cp.Problem(cp.Minimize(self.cost),self.constraints)
        # print('yooooooooooooooooooo')
        print('MPCC\' optimizaton problem successfully created.')
        assert self.problem.is_dcp(dpp=True)

    ########
    ### Make list of parameters
    def MakeParaList(self,N,*args,**kwargs):
        para=[]
        for k in range(N):
            para +=[cp.Parameter(*args,**kwargs)]
        return para

    ### Make list of Variables
    def MakeVarList(self,N,*args,**kwargs):
        var=[]
        for k in range(N):
            var +=[cp.Variable(*args,**kwargs)]
        return var

    ### List values to list parameter values
    def List2Para(self, Para_list,Val_list):
        n = len(Para_list)
        assert n==len(Val_list), 'length differents'
        for k in range(n):
            Para_list[k].value = Val_list[k]
        return Para_list

    ### list parameter values to list values
    def Para2List(self,Para_list):
        n = len(Para_list)
        Val_list = []
        for k in range(n):
            Val_list += [Para_list[k].value]
        return Val_list
    
    def Para2Para(self,ParaSource,ParaDest):
        n = len(ParaSource)
        assert len(ParaDest) == n
        for k in range(n):
            ParaDest[k].value = ParaSource[k].value
        return ParaDest
    #########
    
    
    def UpdateNominalData(self): #(doesnt accept vectorized )
        # [[X,Y,theta,dX,dY,dtheta],delta,D1,D2,C1,C2,B1,B2,a1,a2]
        
        for k in range(self.N):
            qk = self._q[k].value
            uk = self._u[k].value
            
            Ak,Bk,Gk = DiscDynaMatrices(self.dt,qk.squeeze(),uk.squeeze(),self.M,self.I,self.a1,self.a2,self.D1,self.D2,self.C1,self.C2,self.B1,self.B2)
            
            
            qkp1 = (Ak@qk + Bk@uk + Gk)
            
            self._q[k+1].value, self.A_[k].value, self.B_[k].value, self.G_[k].value = qkp1,Ak,Bk,Gk

            tak = self._ta[k].value
            vk  = self._v[k].value
            takp1 = tak + vk*self.dt 
            # # # error at 0 is fixed. Not considering it. saving error matrices from k =1  at position 0
            
            Xref,dXref,ddXref = GetBezierOut(self.alphaX,takp1[0,0],True,True)
            Yref,dYref,ddYref = GetBezierOut(self.alphaY,takp1[0,0],True,True)
            Cekp1,Gradeqkp1,Gradetakp1 = ErrorsLindata(qkp1.squeeze(),takp1.squeeze(),Xref,Yref,dXref,dYref,ddXref,ddYref) #accepts np.array
            Grade = np.concatenate((Gradeqkp1,Gradetakp1),1)
            # # # e_lin=e(nom)+grad([q;ta]-nom)=> e_lin'We_lin=> const+[q ta]Grad'WGrad[q;ta]+2e(nom)WGrad
            
            self.Ge[k].value = 2*Cekp1.transpose()@self.We@Grade
            W,V = np.linalg.eig(Grade.transpose()@self.We@Grade)
            self.Ae[k].value = V@np.diag(np.sqrt(np.abs(np.real(W))))@np.linalg.inv(V)
            self._ta[k+1].value = takp1
            # self._Phi[k].value = np.array([[np.arctan2(dYref,dXref)]])

            # # #boundary
            self.a_i[k].value,self.b_i[k].value,self.c_i[k].value,self.a_o[k].value,self.b_o[k].value,self.c_o[k].value=GetTrackBoundsConstants(self._q[k+1].value[0],self._q[k+1].value[1],self.xc,self.yc,self.xi,self.yi,self.xo,self.yo)
            
            # slip angle
            C_alpha,G_alpha = AlphaLindata(qkp1.squeeze(),self.a1,self.a2)
            self.G_alpha[k].value = 2*C_alpha.transpose()@self.Q_alpha@G_alpha
            self.A_alpha[k].value = G_alpha.transpose()@self.Q_alpha@G_alpha


    def GetNominalPath(self):
        Xnom,Ynom=[],[]
        for k in range(self.N):
            Xnom += [self._q[k].value[0][0]]
            Ynom += [self._q[k].value[1][0]]
        return Xnom, Ynom

    def GetBezierNominalPath(self):
        t = np.array([t.value[0][0] for t in self._ta])
        BXnom = GetBezierOut(self.alphaX,t)
        BYnom = GetBezierOut(self.alphaY,t)
        return BXnom, BYnom

    def SetInitialCondition(self,q0,ta0,alphaX,alphaY):
        self.alphaX = alphaX
        self.alphaY = alphaY
        self._ta[0].value = ta0.reshape((1,1))
        self._q[0].value = q0.reshape((self.N_states,1))
        # print(q0)
        
        self.UpdateNominalData()


    def SolveMPCC(self): #solves one iteration of mpcc
        self.problem.solve(verbose = False,solver='OSQP')
        
    def ShiftOneStep(self,Para):
        temp = self.Para2List(Para)
        temp.pop(0)
        temp = temp + [temp[-1]]
        Para = self.List2Para(Para,temp)
        return Para
    
    def SequentiallySolveMPCC(self,MaxIter,dq_thresh):
        ITERATE = True
        i       = 0
        # self._u = self.ShiftOneStep(self._u) #warmstart
        # self._v = self.ShiftOneStep(self._v)
        while ITERATE:
            tmp = time.time()
            self.UpdateNominalData()
            
            self.SolveMPCC()
            maxdiff = np.linalg.norm([np.linalg.norm(_q[0:2]-q[0:2],2) for _q,q in zip(self.Para2List(self._q),self.Para2List(self.q))],2)
            self._q     = self.Para2Para(self.q,self._q)
            self._u     = self.Para2Para(self.u,self._u)
            self._v     = self.Para2Para(self.v,self._v)
            # print(maxdiff,i)
            if i > MaxIter:
                ITERATE = False
                # print("Max iteration reached")
            elif dq_thresh > maxdiff:
                ITERATE = False
                # print("q threshold reached")
            else:
                i += 1
            # print(max(self.Para2List(self.s_i)),min(self.Para2List(self.s_i)))
                 
            

