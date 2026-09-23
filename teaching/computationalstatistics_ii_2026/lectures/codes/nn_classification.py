"""Generate the two-dimensional and synthetic-image classification examples.
Run manually with NumPy and Matplotlib; Quarto only displays saved results.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

BASE=Path(__file__).resolve().parents[1]
BLUE,RED,PURPLE='#0f4494','#be2832','#7d3c98'
plt.rcParams.update({'font.size':13,'svg.fonttype':'none'})


def sigmoid(s):
    return np.exp(-np.logaddexp(0,-s))


def fit(p, forward, X, y, Xv, yv, epochs, rate):
    m,v=[np.zeros_like(a) for a in p],[np.zeros_like(a) for a in p]
    best_loss=np.inf
    history=[]
    for t in range(1,epochs+1):
        s,grads=forward(p,X,y)
        for j,g in enumerate(grads):
            m[j]=.9*m[j]+.1*g; v[j]=.999*v[j]+.001*g*g
            p[j]-=rate*(m[j]/(1-.9**t))/(np.sqrt(v[j]/(1-.999**t))+1e-8)
        sv,_=forward(p,Xv)
        loss=np.mean(np.logaddexp(0,sv)-yv*sv)
        history.append(loss)
        if loss<best_loss:
            best_loss,best_epoch=loss,t
            best=[a.copy() for a in p]
    return best,best_epoch,history


def mlp(p,X,y=None):
    W,b,v,c=p
    H=np.tanh(X@W+b); s=H@v+c
    if y is None:return s,None
    ds=(sigmoid(s)-y)/len(y)
    dz=ds[:,None]*v*(1-H*H)
    return s,[X.T@dz,dz.sum(0),H.T@ds,np.asarray(ds.sum())]


def plane():
    rng=np.random.default_rng(314)
    def sample(n):
        X=rng.uniform(-1,1,(n,2))
        y=rng.binomial(1,sigmoid(12*(np.sum(X*X,axis=1)-.5)))
        return X,y
    X,y=sample(300);Xv,yv=sample(150);Xt,yt=sample(500)
    p=[rng.normal(0,.7,(2,16)),np.zeros(16),rng.normal(0,.25,16),np.array(0.)]
    p,epoch,_=fit(p,mlp,X,y,Xv,yv,1200,.02)
    s,_=mlp(p,Xt); accuracy=float(np.mean((s>=0)==yt))
    a=np.linspace(-1,1,160);xx,yy=np.meshgrid(a,a)
    scores,_=mlp(p,np.c_[xx.ravel(),yy.ravel()])
    fig,ax=plt.subplots(figsize=(6,4.6),layout='constrained')
    plot=ax.contourf(xx,yy,sigmoid(scores).reshape(xx.shape),levels=np.linspace(0,1,11),cmap='RdBu_r',alpha=.7)
    ax.contour(xx,yy,scores.reshape(xx.shape),levels=[0],colors=[PURPLE],linewidths=2)
    ax.scatter(X[:,0],X[:,1],c=y,cmap='RdBu_r',vmin=0,vmax=1,s=13,edgecolors='white',linewidths=.3)
    ax.add_patch(plt.Circle((0,0),np.sqrt(.5),fill=False,ls='--',color='black'))
    ax.set(xlabel='$x_1$',ylabel='$x_2$',aspect='equal')
    fig.colorbar(plot,ax=ax,label='Predicted P(Y=1 | x)')
    fig.savefig(BASE/'images/nn-classification.svg',transparent=True);plt.close(fig)
    return dict(selected_epoch=epoch,test_accuracy=accuracy)


def cnn(p,X,y=None):
    K,b,v,c=p
    # X contains all valid 3x3 patches, shape (batch,6,6,3,3).
    Z=np.einsum('nijuv,uvc->nijc',X,K)+b
    H=np.maximum(Z,0); pooled=H.mean(axis=(1,2));s=pooled@v+c
    if y is None:return s,None
    ds=(sigmoid(s)-y)/len(y)
    dz=ds[:,None,None,None]*v*(Z>0)/36
    return s,[np.einsum('nijuv,nijc->uvc',X,dz),dz.sum(axis=(0,1,2)),pooled.T@ds,np.asarray(ds.sum())]


def images():
    rng=np.random.default_rng(2718)
    def sample(n):
        y=rng.integers(0,2,n)
        X=rng.normal(0,.3,(n,8,8))
        for i in range(n):
            pos=rng.integers(1,6)
            if y[i]==0:X[i,pos:pos+2,:]+=1
            else:X[i,:,pos:pos+2]+=1
        return X,y
    X,y=sample(200);Xv,yv=sample(100);Xt,yt=sample(200)
    patches=lambda a:sliding_window_view(a,(3,3),axis=(1,2))
    p=[rng.normal(0,np.sqrt(2/9),(3,3,4)),np.zeros(4),rng.normal(0,.5,4),np.array(0.)]
    # Independent finite-difference check for every parameter, away from ReLU kinks.
    tiny=patches(X[:3]);_,grads=cnn(p,tiny,y[:3]);errors=[]
    loss=lambda:np.mean(np.logaddexp(0,cnn(p,tiny)[0])-y[:3]*cnn(p,tiny)[0])
    for a,g in zip(p,grads):
        for idx in np.ndindex(a.shape):
            old=a[idx].copy();a[idx]=old+1e-6;plus=loss()
            a[idx]=old-1e-6;minus=loss();a[idx]=old
            errors.append(abs((plus-minus)/2e-6-g[idx]))
    assert max(errors)<1e-7,max(errors)
    p,epoch,history=fit(p,cnn,patches(X),y,patches(Xv),yv,250,.02)
    scores,_=cnn(p,patches(Xt));pred=(scores>=0).astype(int)
    accuracy=float(np.mean(pred==yt))
    fig,axs=plt.subplots(2,4,figsize=(10,4.2),layout='constrained')
    for i,ax in enumerate(axs.flat):
        ax.imshow(Xt[i],cmap='gray',vmin=0,vmax=1)
        ax.set_title(f"True: {'V' if yt[i] else 'H'}; predicted: {'V' if pred[i] else 'H'}",fontsize=11)
        ax.axis('off')
    fig.savefig(BASE/'images/nn-cnn-predictions.svg',transparent=True);plt.close(fig)
    return dict(selected_epoch=epoch,test_accuracy=accuracy,correct=int(np.sum(pred==yt)),test_size=len(yt),gradient_check_error=float(max(errors)))


if __name__=='__main__':
    results=dict(plane=plane(),cnn=images())
    (BASE/'codes/nn_classification_results.json').write_text(json.dumps(results,indent=2)+'\n')
    (BASE/'codes/nn_plane_results.qmd').write_text(f"Selected epoch: **{results['plane']['selected_epoch']}**; test accuracy: **{results['plane']['test_accuracy']:.1%}** on 500 independent observations.\n")
    r=results['cnn']
    (BASE/'codes/nn_cnn_results.qmd').write_text(f"Selected epoch: **{r['selected_epoch']}**; test accuracy: **{r['test_accuracy']:.1%}** ({r['correct']}/{r['test_size']}). This synthetic task is deliberately simple.\n")
    print(json.dumps(results,indent=2))
