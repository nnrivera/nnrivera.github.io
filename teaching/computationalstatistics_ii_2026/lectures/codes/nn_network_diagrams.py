"""Draw consistent vector notation for neural-network slides.

Run manually with MPLCONFIGDIR=/tmp/nn-mpl python3 lectures/codes/nn_network_diagrams.py.
Shapes are a convention for this lecture, reinforced by the activation curve.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Polygon

OUT=Path(__file__).resolve().parents[1]/'images'
BLUE, PURPLE, RED, INK, GREY = '#0f4494','#7d3c98','#be2832','#232754','#94a3b8'
plt.rcParams.update({'font.size':14,'svg.fonttype':'none','font.family':'DejaVu Sans'})


def canvas(size, limits):
    fig,ax=plt.subplots(figsize=size,layout='constrained')
    ax.set(xlim=limits[:2],ylim=limits[2:],aspect='equal')
    ax.axis('off')
    return fig,ax


def save(fig,name):
    fig.savefig(OUT/name,transparent=True)
    plt.close(fig)


def text(ax,x,y,label,size=14,colour=INK,**kw):
    return ax.text(x,y,label,ha='center',va='center',fontsize=size,color=colour,**kw)


def edge(ax,start,end,label=None,colour=BLUE,dashed=False,reverse=False,offset=(0,0),shrink=21):
    if reverse:start,end=end,start
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,
                                shrinkA=shrink,shrinkB=shrink,color=colour,lw=1.6,
                                linestyle='--' if dashed else '-'))
    if label:
        x=(start[0]+end[0])/2+offset[0];y=(start[1]+end[1])/2+offset[1]
        text(ax,x,y,label,13,colour,bbox=dict(facecolor='white',edgecolor='none',pad=1.5))


def value_node(ax,xy,label,constant=False,r=.30):
    x,y=xy
    if constant:
        ax.add_patch(Polygon([(x,y+r),(x+r,y),(x,y-r),(x-r,y)],closed=True,
                             facecolor='#fff1f2',edgecolor=RED,lw=2))
    else:
        ax.add_patch(Circle(xy,r,facecolor='white',edgecolor=BLUE,lw=2))
    text(ax,x,y,label,16,RED if constant else BLUE)


def activation(ax,xy,kind='relu',label=None,size=.76,inactive=False):
    x,y=xy;r=size/2;colour=GREY if inactive else PURPLE
    fill='#f1f5f9' if inactive else '#fbf8fd'
    if kind=='tanh':
        vertices=[(x-r*.62,y-r),(x+r*.62,y-r),(x+r,y),(x+r*.62,y+r),(x-r*.62,y+r),(x-r,y)]
        ax.add_patch(Polygon(vertices,closed=True,facecolor=fill,edgecolor=colour,lw=2))
    else:
        radius=size*.28 if kind=='sigmoid' else .035
        ax.add_patch(FancyBboxPatch((x-r,y-r),size,size,boxstyle=f'round,pad=0,rounding_size={radius}',
                                   facecolor=fill,edgecolor=colour,lw=2))
    z=np.linspace(-3,3,61);u=x+z/3*r*.67
    if kind=='relu':
        v=y+np.maximum(z,0)/3*r*1.05-r*.45
        zero=y-r*.45
    elif kind=='sigmoid':
        v=y+(1/(1+np.exp(-z))-.5)*r*1.35
        zero=y-r*.675
    else:
        v=y+np.tanh(z)*r*.62
        zero=y
    ax.plot([x-r*.76,x+r*.76],[zero,zero],color='#cbd5e1',lw=.8,zorder=3)
    ax.plot([x,x],[y-r*.76,y+r*.76],color='#cbd5e1',lw=.8,zorder=3)
    ax.plot(u,v,color=colour,lw=2,zorder=4)
    if label:text(ax,x,y-r-.22,label,13,INK)


def legend():
    fig,ax=canvas((12,3.5),(-.8,11.8,-1.45,1.6))
    for x,kind,title,range_label in [(1,'relu','ReLU','Zero below 0; then linear'),(5.5,'sigmoid','Sigmoid','From 0 to 1; midpoint 1/2'),(10,'tanh','tanh','From −1 to 1; midpoint 0')]:
        activation(ax,(x,0),kind,size=1.15)
        text(ax,x,1.15,title,19)
        text(ax,x,-1.03,range_label,13)
    save(fig,'nn-activation-symbols.svg')


def neuron():
    fig,ax=canvas((10,3.7),(-.7,9.7,-1.65,1.85))
    inputs=[(0,.4),(0,-.7)];summation=(3,0);act=(6,0);output=(9,0)
    for j,pt in enumerate(inputs,1):
        edge(ax,pt,summation,rf'$w_{j}$',offset=(0,.12 if j==1 else -.12))
        value_node(ax,pt,rf'$x_{j}$')
    const=(1.7,1.45);edge(ax,const,summation,'$b$',RED,True,offset=(.2,.1));value_node(ax,const,'1',True)
    value_node(ax,summation,r'$\Sigma$')
    edge(ax,summation,act,'$z$',offset=(0,.25));activation(ax,act,'relu',size=.9)
    edge(ax,act,output);value_node(ax,output,'$h$')
    text(ax,3,-1.1,'Weighted sum\nand intercept',13)
    text(ax,6,-1.1,'Activation\nReLU shown',13)
    text(ax,9,-1.1,'Output',13)
    save(fig,'nn-neuron.svg')


def network(filename='nn-one-hidden-layer.svg',forward=False,backward=False):
    fig,ax=canvas((8.8,5.2),(-.65,7.1,-2.35,2.95))
    inp=[(0,.7),(0,-.9)];hidden=[(3,1.1),(3,-.15),(3,-1.4)];out=(6.5,-.15)
    for i,start in enumerate(inp):
        for j,end in enumerate(hidden):
            edge(ax,start,end,colour=GREY if (forward or backward) and j==0 else BLUE,
                 reverse=backward)
    for j,start in enumerate(hidden):
        if backward:
            lab=[r'$-3$',r'$3/2$',r'$-3/2$'][j]
            colour=PURPLE
            edge(ax,start,out,lab,colour,reverse=True,offset=(.15,0))
        else:
            lab=[r'$2\times0=0$',r'$-1\times2=-2$',r'$1\times1=1$'][j] if forward else rf'$v_{j+1}$'
            edge(ax,start,out,lab,GREY if forward and j==0 else PURPLE,offset=(.25,0))
    for i,pt in enumerate(inp):value_node(ax,pt,rf'$x_{i+1}$' if not(forward or backward) else str(i+1))
    for j,pt in enumerate(hidden):
        activation(ax,pt,'relu',size=.66,inactive=(forward or backward) and j==0)
        if forward:
            text(ax,pt[0],pt[1]+.51,rf'$z_{j+1}={[-1,2,1][j]}$',12)
            text(ax,pt[0],pt[1]-.5,rf'$h_{j+1}={ [0,2,1][j]}$',12)
        elif backward:
            text(ax,pt[0],pt[1]-.48,[r'$\delta_{z,1}=0$',r'$\delta_{z,2}=3/2$',r'$\delta_{z,3}=-3/2$'][j],11,GREY if j==0 else INK)
        else:text(ax,pt[0],pt[1]-.5,rf'$h_{j+1}$',14)
    value_node(ax,out,r'$\Sigma$')
    if forward:text(ax,6.5,-.9,r'$s=-1/2$',15)
    elif backward:text(ax,6.5,-.85,r'$\delta_s=-3/2$',14)
    else:text(ax,6.5,-.8,'$s$',16)
    constants=[(0,2.35),(3,2.35)]
    for pt in constants:value_node(ax,pt,'1',True,r=.26)
    if not backward:
        for j,target in enumerate(hidden):
            lab=rf'$b_{j+1}$' if not forward else rf'$b_{j+1}={ [0,-1,2][j]}$'
            edge(ax,constants[0],target,lab,RED,True,offset=(-.38,.0),shrink=23)
        edge(ax,constants[1],out,'$c$' if not forward else r'$c=1/2$',RED,True,offset=(.23,.18))
    else:
        # Constants are shown even in a backward diagram; their outgoing parameters are intercepts.
        for j,target in enumerate(hidden):edge(ax,constants[0],target,colour=RED,dashed=True,shrink=23)
        edge(ax,constants[1],out,colour=RED,dashed=True)
        text(ax,1.1,2.7,'Intercept inputs',11,RED)
    if not(forward or backward):
        text(ax,1.3,-2.13,r'Input weights $w_{jk}$',13,BLUE)
        text(ax,4.9,-2.13,r'Output weights $v_j$',13,PURPLE)
    elif backward:
        text(ax,3.4,-2.15,'Reverse arrows carry derivatives; the ReLU gates use stored forward signs.',11)
    else:text(ax,3.4,-2.15,'The first ReLU is inactive for this input.',12)
    save(fig,filename)


def deep():
    fig,ax=canvas((12,4.1),(-.6,11.2,-2.15,2.1))
    layers=[[(0,.6),(0,-.6)],[(3,1),(3,0),(3,-1)],[(6.4,.65),(6.4,-.65)],[(10,0)]]
    for index,(left,right) in enumerate(zip(layers,layers[1:])):
        for a in left:
            for b in right:edge(ax,a,b,colour=[BLUE,PURPLE,BLUE][index],shrink=19)
        const=(left[0][0],1.8);value_node(ax,const,'1',True,r=.21)
        for b in right:edge(ax,const,b,colour=RED,dashed=True,shrink=20)
    for j,pt in enumerate(layers[0],1):value_node(ax,pt,rf'$x_{j}$',r=.24)
    for pt in layers[1]:activation(ax,pt,'relu',size=.61)
    for pt in layers[2]:activation(ax,pt,'tanh',size=.61)
    value_node(ax,layers[3][0],r'$\Sigma$',r=.28)
    for x,title,subtitle in [(0,'Input','2 coordinates'),(3,'ReLU layer','width 3'),(6.4,'tanh layer','width 2'),(10,'Affine score','$s$')]:
        text(ax,x,-1.5,title,14);text(ax,x,-1.85,subtitle,12)
    for x,title in [(1.5,r'$W^{(1)},b^{(1)}$'),(4.7,r'$W^{(2)},b^{(2)}$'),(8.2,r'$W^{(3)},b^{(3)}$')]:
        text(ax,x,1.65,title,13)
    save(fig,'nn-deep-network.svg')


def output_heads():
    fig,ax=canvas((12,4.2),(-.6,11.5,-2.05,2.05))
    rows=[(1.35,'Regression','identity'),(0,'Binary classification','sigmoid'),(-1.35,'K-class classification','softmax')]
    for y,title,kind in rows:
        text(ax,.8,y,title,14)
        text(ax,2.4,y,'$h$',15)
        edge(ax,(2.4,y),(4.25,y));value_node(ax,(4.25,y),r'$\Sigma$')
        text(ax,3.8,y+.48,'Affine scores',10)
        edge(ax,(4.25,y),(6.9,y),'$s$' if kind!='softmax' else r'$s_1,\ldots,s_K$',offset=(0,.1))
        if kind=='sigmoid':
            activation(ax,(6.9,y),'sigmoid',size=.62)
            text(ax,9.5,y,r'$p\in(0,1)$',17)
        elif kind=='identity':
            text(ax,6.9,y,'No activation',12)
            text(ax,9.5,y,r'$s\in\mathbb{R}$',17)
        else:
            ax.add_patch(FancyBboxPatch((6.25,y-.31),1.3,.62,boxstyle='round,pad=.02',facecolor='#f5eff8',edgecolor=PURPLE,lw=2))
            text(ax,6.9,y,'softmax',12,PURPLE)
            text(ax,9.65,y,'Probabilities\nsumming to 1',12)
        edge(ax,(7.75,y),(8.7,y),colour=PURPLE,shrink=0)
        value_node(ax,(3.3,y+.48),'1',True,r=.14)
        edge(ax,(3.3,y+.48),(4.25,y),colour=RED,dashed=True,shrink=13)
    save(fig,'nn-output-heads.svg')


def learning_graph():
    fig,ax=canvas((12,3.8),(-.7,12.7,-1.65,1.55))
    for a,b in [((0,0),(2.7,0)),((2.7,0),(5.5,0)),((5.5,0),(8.2,0)),((8.2,0),(11.6,0))]:edge(ax,a,b)
    value_node(ax,(0,0),'$x$');value_node(ax,(2.7,0),r'$\Sigma$')
    activation(ax,(5.5,0),'relu',size=.82)
    value_node(ax,(8.2,0),r'$\Sigma$')
    ax.add_patch(FancyBboxPatch((10.8,-.4),1.6,.8,boxstyle='round,pad=0.03',facecolor='#eef4fb',edgecolor=BLUE,lw=2))
    text(ax,11.6,0,'Loss',16)
    for x,label in [(2.7,'$z$'),(5.5,'$h$'),(8.2,'$s$')]:text(ax,x,-.68,label,16)
    for const,target,label in [((1.6,1.15),(2.7,0),'$b$'),((7.1,1.15),(8.2,0),'$c$')]:
        value_node(ax,const,'1',True,r=.23);edge(ax,const,target,label,RED,True,offset=(.2,0))
    text(ax,1.35,.3,'$w$',15,BLUE);text(ax,6.85,.3,'$v$',15,BLUE)
    edge(ax,(11.6,1.05),(11.6,0),'$y$',RED,offset=(.45,0),shrink=16)
    for a,b,label in [(11.6,8.2,'Start: score − response'),(8.2,5.5,'Multiply by output weight'),(5.5,2.7,'Pass through ReLU gate')]:
        edge(ax,(a,-1.15),(b,-1.15),colour=PURPLE,shrink=5)
        text(ax,(a+b)/2,-1.47,label,10,PURPLE)
    save(fig,'nn-computation-graph.svg')


def generate():
    legend();neuron();network();network('nn-forward-pass.svg',forward=True)
    network('nn-backward-pass.svg',backward=True);deep();output_heads();learning_graph()


if __name__=='__main__':generate()
