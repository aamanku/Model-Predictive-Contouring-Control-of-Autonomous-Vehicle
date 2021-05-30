'''
Written By: Abhijeet Kulkarni (amkulk@udel.edu)
For: Research Project MEEG678
Title: Autonomous driving using Model Predictive ContouringControl.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from functions.Car_Dynamics import *
from functions.HelpingFunctions import *
from functions.MPCCFunctions import *

# %matplotlib qt
fig, ax = plt.subplots(figsize=(15,15)) # You may need to adjust the figsize to fit your screen
# Draw the road
L = 50.0     # length of straight section [m]
R = 5.0     # radius of curved section centerline [m]
w = 2     # width of lane [m]
Ri = R-w    # radius of inner boundary [m]
Ro = R+w    # radius of outer boundary [m]
ds = 0.5   # step size [m]

# (xc,yc) = coordinates of center dotted line
# (xi,yi) = coordinates of inner solid line
# (xo,yo) = coordinates of outer solid line
# xc = np.hstack((np.arange(0,L,ds), L+R*np.cos(np.arange(-np.pi/2,np.pi/2,ds/R)),
#                np.arange(L,0,-ds), R*np.cos(np.arange(np.pi/2,3*np.pi/2,ds/R))))
# yc = np.hstack((np.zeros_like(np.arange(0,L,ds)), R+R*np.sin(np.arange(-np.pi/2,np.pi/2,ds/R)),
#                np.full_like(np.arange(L,0,-ds),2*R), R+R*np.sin(np.arange(np.pi/2,3*np.pi/2,ds/R))))
xc = np.hstack((np.arange(0,L,ds), L+R*np.cos(np.arange(-np.pi/2,np.pi/2,ds/R)),
               np.arange(L,0,-ds), R*np.cos(np.arange(np.pi/2,3*np.pi/2,ds/R))))
yc = np.hstack((np.zeros_like(np.arange(0,L,ds)), R+R*np.sin(np.arange(-np.pi/2,np.pi/2,ds/R)),
               np.full_like(np.arange(L,0,-ds),2*R), R+R*np.sin(np.arange(np.pi/2,3*np.pi/2,ds/R))))
w_i,w_o = w*np.ones_like(xc),-w*np.ones_like(xc)
w_i[50:60] = -0.5
w_o[80:90] = -0.5
w_o[100:120] = -1
w_i[160:170] = -0.75
# w_i[240:250] = 0
w_o[220:225] = 0
xi,yi=GenOffsetTrack(xc,yc,w_i)
xo,yo=GenOffsetTrack(xc,yc,w_o)


plt.plot(xc,yc,color='b',dashes=[20, 6])

plt.plot(xi,yi,color='k')
plt.plot(xo,yo,color='k')
ax.set_aspect('equal')
ax.axis('off')
# plt.show()
# fig.savefig('Obstacles')


t_max = 12.0   # final time of the simulat

dt = 0.005
ini_states = np.array([5,1,0,1,0,0])
CarD=CarDynamics(ini_states,dt)
print('Generated car object.')
# Draw the Duckiebot
Scale = 20 #43
l_car = (CarD.a1 + CarD.a2)*Scale
w_car = (l_car/3)
r_cam = 0.4     # camera range [m]
phi_cam = 45.0  # camera angle half-width [deg]
car = mpatches.Rectangle([-CarD.a2*Scale,-w_car/2],l_car,w_car,linewidth=1,edgecolor='k',facecolor='r')
ax.add_patch(car)
cam_view = mpatches.Wedge((CarD.a1,0),r_cam,-phi_cam,phi_cam,facecolor='yellow',edgecolor=None)
ax.add_patch(cam_view)

text = plt.annotate('',(L/1.5-0.25,R),fontsize=8,animated=True)
path, = ax.plot([],[],color = 'r',linewidth=2,alpha = 1, aa = True,label = 'Local track')
clpoint, = ax.plot([],[],color = 'k',marker='o',linewidth=5,alpha = 1, aa = True,label='Closest point')
nomialpath, = ax.plot([],[],color = 'k',linewidth=1,alpha = 1, aa = True,label='Predicted by MPCC')
bezierpath, = ax.plot([],[],color = 'g',linewidth=1,alpha = 1, aa = True,label='Predicted progress')
truetrajectory, = ax.plot([],[],color = 'm', linewidth = 2, alpha = 0.75, aa = True, label = 'True COM trajectory')
TruetrajX,TruetrajY = [ini_states[0]],[ini_states[1]]

def GetLocalBezierXY(Xcar,Ycar,length,beforelen,degree,numpts):
    global xc,yc,ds
    Maxidx = len(xc)
    Startidx = findclosestONTrack(Xcar,Ycar,xc,yc) - int(beforelen/ds)
    Endidx = Startidx + int(length/ds)
    if Startidx%Maxidx > Endidx%Maxidx:
        pointsxc = np.append(xc[Startidx%Maxidx::],xc[0:Endidx%Maxidx])
        pointsyc = np.append(yc[Startidx%Maxidx::],yc[0:Endidx%Maxidx])
    else:
        pointsxc = xc[Startidx%Maxidx:Endidx%Maxidx]
        pointsyc = yc[Startidx%Maxidx:Endidx%Maxidx]
    t_points = np.linspace(0,1,len(pointsxc))
    alphaX = fitCurvetopoints(pointsxc,t_points,degree)
    alphaY = fitCurvetopoints(pointsyc,t_points,degree)
    t = np.linspace(0,1,numpts)
    tc = ProjectOnBCurveXY(np.array([alphaX,alphaY]),[Xcar,Ycar])
    Xvec = GetBezierOut(alphaX,np.append(t,tc))
    Yvec = GetBezierOut(alphaY,np.append(t,tc))
    return Xvec[0:-1],Yvec[0:-1],Xvec[-1],Yvec[-1],alphaX,alphaY,tc #refpath bez, closestpoint on bez

inputs=[0,0,0]
fps = 30
N_mpcc = 30 #horizon length of the MPC
L_LocalTrack = 20

mpcc = MPCC_Object(N_mpcc,N_mpcc/fps,xc,yc,xi,yi,xo,yo)
mpcc.SetInitialCondition(CarD.states,np.array([0]),[0,2,3,4,5,20],[0,0,0,0,0,0])
print('Initilization Complete.')
print('Starting Simulation.')
# Initialize the axes for animation
def init():
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend()
    
    return (car,cam_view,text,path,clpoint,nomialpath,bezierpath,truetrajectory)

def update(t):
    global CarD,mpcc

    x,y,tht,_,_,_=CarD.states
    car.set_transform(Affine2D().rotate(tht) + Affine2D().translate(x,y) + ax.transData)
    cam_view.set_transform(Affine2D().rotate(tht) + Affine2D().translate(x,y) + ax.transData)

    Xbez,Ybez,Xcen,Ycen,alphaX,alphaY,tc= GetLocalBezierXY(x,y,L_LocalTrack,L_LocalTrack/10,7,50)
    
    path.set_data(Xbez,Ybez)
    clpoint.set_data(Xcen,Ycen)


    mpcc.SetInitialCondition(CarD.states,tc,alphaX,alphaY)
    # mpcc.SequentiallySolveMPCC(20,1)
    try: 
        mpcc.SequentiallySolveMPCC(20,1)#(num_iterations,changethreshold)
    except:
        pass
    
    Xnom,Ynom = mpcc.GetNominalPath()
    nomialpath.set_data(Xnom,Ynom)

    BXnom,BYnom = mpcc.GetBezierNominalPath()
    bezierpath.set_data(BXnom,BYnom)

    inputs = list(mpcc._u[0].value.squeeze())
    for _ in range(0,int(1//(fps*CarD.dt))):
        CarD.sim_step(inputs)
    TruetrajX.append(CarD.states[0])
    TruetrajY.append(CarD.states[1])
    truetrajectory.set_data(TruetrajX,TruetrajY)
    text.set_text(f't:{t:.1f} sec\n Steering: {np.rad2deg(inputs[0]):0.1f} deg \n Thrust: {inputs[2]:0.1f} N')
    return (car,cam_view,text,path,clpoint,nomialpath,bezierpath,truetrajectory)

# for _ in range(1000):
#     update(0)
    

ani = FuncAnimation(fig, update, frames=np.linspace(0,t_max,int(t_max*fps)), init_func=init,
                    interval=1000//fps, blit=True, repeat=False)

plt.show()
# plt.close()                 # hide display of final frame
# plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\amkulk\Downloads\ffmpeg-2021-05-19-git-2261cc6d8a-essentials_build\bin\ffmpeg.exe'
# writer = FFMpegWriter(fps=fps)

# ani.save('T20Obstacle.mp4', writer=writer)


# ##saving final trajectory
# fig1, ax1 = plt.subplots(figsize=((L)//2,(2*R+2*w)//2))
# plt.plot(xc,yc,color='b',dashes=[20, 6])
# plt.plot(xi,yi,color='k')
# plt.plot(xo,yo,color='k')
# plt.plot(TruetrajX,TruetrajY,color = 'm', linewidth = 2, alpha = 0.75, aa = True, label = 'COM Trajectory')

# ax1.set_aspect('equal')
# ax1.axis('off')
# fig1.savefig('T20Obstacle')