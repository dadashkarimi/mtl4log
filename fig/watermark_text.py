"""
==============
Text watermark
==============

Use a Text as a watermark
"""
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

# Fixing random state for reproducibility
np.random.seed(19680801)


fig, ax = plt.subplots()

#calendar (den) sing/agg
y1=[0.006,0.006,0.030,0.042,0.065,0.065,0.089,0.065,0.137,0.119]
y2=[0.030,0.006,0.006,0.018,0.083,0.071,0.137,0.137,0.161,0.179]

#publication(den) sing/agg
y3=[0.006,0.087,0.143,0.242,0.106,0.255,0.248,0.304,0.342,0.373]
y4=[0.050,0.137,0.168,0.199,0.174,0.317,0.298,0.307,0.317,0.323]

#calendar(den) m2m/o2m/enc-dec
y5=[0.012,0.012,0.012,0.012,0.060,0.042,0.119,0.083,0.131,0.161]
y6=[0.0238095,0.024,0.030,0.071,0.077,0.048,0.071,0.089,0.143,0.143]
y7=[0.0372671,0.030,0.030,0.054,0.077,0.060,0.060,0.071,0.161,0.214]

#publications(den) m2m/o2m/enc-dec
y8=[0.000,0.000,0.056,0.054,0.060,0.224,0.280,0.280,0.286,0.31]
y9=[0.006,0.050,0.025,0.186,0.236,0.174,0.242,0.311,0.286,0.298]
y10=[0.037,0.118,0.118,0.130,0.193,0.261,0.286,0.298,0.286,0.286]

#calendar(den) enc-dec/k*d
y11=[0.0372671,0.030,0.030,0.054,0.077,0.060,0.060,0.071,0.161,0.214]
y12=[0.012,0.024,0.077,0.024,0.101,0.101,0.179,0.214,0.250,0.274]

#publications(den) enc-dec/k*d
y13=[0.037,0.118,0.118,0.130,0.193,0.261,0.286,0.298,0.286,0.286]
y14=[0.056,0.081,0.199,0.174,0.199,0.298,0.304,0.342,0.435,0.329]

#one out geo/atis/basketball/publication/non #den level
y15=[0.018,0.030, 0.030, 0.018, 0.071, 0.071, 0.167, 0.167, 0.190, 0.220]
y16=[0.006,0.042, 0.048, 0.065, 0.095, 0.101, 0.131, 0.167, 0.304, 0.196]
y17=[0.000,0.030, 0.030, 0.048, 0.089, 0.119, 0.107, 0.179, 0.214, 0.238]
y18=[0.000,0.000, 0.042, 0.065, 0.077, 0.143, 0.071, 0.083, 0.179, 0.167]
y19=[0.054,0.006, 0.030, 0.054, 0.113, 0.125, 0.131, 0.190, 0.196, 0.262]

#one out geo/atis/basketball/publication/non # token level
y20=[0.276,0.424, 0.428, 0.468, 0.511, 0.496, 0.587, 0.561, 0.610, 0.668]
y21=[0.416,0.504, 0.558, 0.597, 0.538, 0.566, 0.600, 0.623, 0.628, 0.685]
y22=[0.296,0.465, 0.359, 0.507, 0.503, 0.507, 0.595, 0.637, 0.650, 0.653]
y23=[0.178,0.440, 0.462, 0.509, 0.521, 0.542, 0.491, 0.562, 0.668, 0.572]
y24=[0.376,0.478, 0.505, 0.472, 0.538, 0.592, 0.615, 0.643, 0.647, 0.621]

x=np.linspace(10, 100, 10) # 

plt.xlabel("Training examples for each task")
plt.ylabel("Accuracy")

#ax.plot(x,y1, '-o', ms=10, lw=2, alpha=0.7, mfc='g',color='g',label='sing')
#ax.plot(x,y2, '-.', ms=10, lw=2, alpha=0.7, mfc='r',color='r',label='agg')
#ax.plot(x,y13, '-o', ms=10, lw=2, alpha=0.7, mfc='b',color='b',label='e2d')
#ax.plot(x,y14, '-o', ms=10, lw=2, alpha=0.7, mfc='k',color='k',label='z-shot k*d')
#ax.plot(x,y10, '-o', ms=10, lw=2, alpha=0.7, mfc='b',color='b',label='e2d')

t, c, k = interpolate.splrep(x, y21, s=0, k=4)
xmin, xmax = x.min(), x.max()
xx = np.linspace(xmin, xmax, 100)
spline = interpolate.BSpline(t, c, k, extrapolate=False)

xnew = np.linspace(x.min(),x.max(),300) 
ax.plot(xx,spline(xx), '-o', ms=1, lw=1, alpha=0.1, mfc='g',color='g',label='-geo')
ax.plot(x,y21, '-o', ms=10, lw=2, alpha=0.7, mfc='r',color='r',label='-atis')
ax.plot(x,y22, ':', ms=10, lw=2, alpha=0.7, mfc='b',color='b',label='-bask')
ax.plot(x,y23, '--', ms=10, lw=2, alpha=0.7, mfc='k',color='k',label='-pub')
ax.plot(x,y24, '-.', ms=10, lw=2, alpha=0.7, mfc='k',color='k',label='all')

ax.legend(loc="best")
ax.grid()

# position bottom right
fig.text(0.95, 0.05, '', fontsize=50, color='gray', ha='right', va='bottom', alpha=0.5)
#fig.set_size_inches(18.5, 10.5)

#plt.show()
plt.savefig('leave-one-out-tok.pdf',format='pdf')
