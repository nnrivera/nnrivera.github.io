"""Reproduce the static neural-network examples; run manually, never from Quarto.

Requires NumPy and Matplotlib. Run:
    MPLCONFIGDIR=/tmp/nn-mpl python3 lectures/codes/nn_experiments.py
All random seeds and candidate settings are fixed before test evaluation.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'images'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 13, 'svg.fonttype': 'none'})
BLUE, RED, PURPLE = '#0f4494', '#be2832', '#7d3c98'


def save(fig, name):
    fig.savefig(OUT / name, transparent=True)
    plt.close(fig)


def truth(x):
    return np.sin(np.pi*x) + 0.3*np.cos(3*np.pi*x)


def predict(p, X):
    W, b, v, c = p
    return np.tanh(X @ W + b) @ v + c


def gradients(p, X, y, penalty):
    W, b, v, c = p
    H = np.tanh(X @ W + b)
    residual = (H @ v + c - y) / len(y)
    delta = residual[:, None] * v * (1-H**2)
    return [X.T @ delta + penalty*W, delta.sum(axis=0),
            H.T @ residual + penalty*v, np.asarray(residual.sum())]


def objective(p, X, y, penalty):
    return 0.5*np.mean((predict(p, X)-y)**2) + penalty/2*(np.sum(p[0]**2)+np.sum(p[2]**2))


def check_gradients():
    rng = np.random.default_rng(7)
    p = [rng.normal(size=(2,3)), rng.normal(size=3), rng.normal(size=3), np.array(0.1)]
    X, y = rng.normal(size=(4,2)), rng.normal(size=4)
    analytic = gradients(p, X, y, 0.03)
    errors = []
    for parameter, grad in zip(p, analytic):
        for idx in np.ndindex(parameter.shape):
            old = parameter[idx].copy()
            parameter[idx] = old + 1e-6
            plus = objective(p, X, y, 0.03)
            parameter[idx] = old - 1e-6
            minus = objective(p, X, y, 0.03)
            parameter[idx] = old
            errors.append(abs((plus-minus)/2e-6-grad[idx]))
    assert max(errors) < 1e-7, max(errors)
    return float(max(errors))


def train(X, y, Xval, yval, width, penalty, epochs=1500):
    rng = np.random.default_rng(1200+width)
    p = [rng.normal(0, 1, (1,width)), np.zeros(width),
         rng.normal(0, 1/np.sqrt(width), width), np.array(0.)]
    m, v = [np.zeros_like(a) for a in p], [np.zeros_like(a) for a in p]
    history, best, best_error, step = [], None, np.inf, 0
    for epoch in range(1, epochs+1):
        indices = rng.permutation(len(y))
        for start in range(0, len(y), 20):
            batch = indices[start:start+20]
            grads = gradients(p, X[batch], y[batch], penalty)
            step += 1
            for j, g in enumerate(grads):
                m[j] = .9*m[j] + .1*g
                v[j] = .999*v[j] + .001*g*g
                p[j] -= .01*(m[j]/(1-.9**step))/(np.sqrt(v[j]/(1-.999**step))+1e-8)
        mse = [np.mean((predict(p,X)-y)**2), np.mean((predict(p,Xval)-yval)**2)]
        history.append(mse)
        if mse[1] < best_error:
            best_error, best_epoch = mse[1], epoch
            best = [a.copy() for a in p]
    return dict(width=width, penalty=penalty, final=p, best=best,
                epoch=best_epoch, history=np.asarray(history), val=float(best_error))


def regression():
    rng = np.random.default_rng(2026)
    def sample(n):
        x = rng.uniform(-1,1,n)
        return x, truth(x)+rng.normal(0,.2,n)
    x, y = sample(60)
    xv, yv = sample(120)
    xt, yt = sample(500)
    mean, scale = x.mean(), x.std()
    transform = lambda a: ((a-mean)/scale)[:,None]
    X, Xv = transform(x), transform(xv)
    runs = [train(X,y,Xv,yv,w,lam) for w in [4,32] for lam in [0.,.01]]
    selected = min(runs,key=lambda r:r['val'])
    # The test set is evaluated only for the selected checkpoint.
    test_mse = float(np.mean((predict(selected['best'],transform(xt))-yt)**2))
    grid = np.linspace(-1,1,500)
    def fitplot(ax,p,label):
        ax.scatter(x,y,s=17,color='#64748b',alpha=.65,label='Training data')
        ax.plot(grid,truth(grid),color=RED,ls='--',label='True mean')
        ax.plot(grid,predict(p,transform(grid)),color=BLUE,label=label)
        ax.set(xlabel='x',ylabel='Response',ylim=(-1.7,1.7))
    baseline = next(r for r in runs if r['width']==32 and r['penalty']==0)
    fig, axs = plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for j,label in enumerate(['Training','Validation']):
        axs[0].plot(np.arange(1,1501),baseline['history'][:,j],label=label,color=[BLUE,PURPLE][j])
    axs[0].set(xlabel='Epoch',ylabel='MSE',yscale='log')
    axs[0].legend(frameon=False)
    fitplot(axs[1],baseline['final'],'Final fit')
    axs[1].legend(frameon=False,fontsize=10)
    save(fig,'nn-training.svg')
    fig, ax = plt.subplots(figsize=(6.5,4.4),layout='constrained')
    ax.scatter(x,y,s=16,color='#64748b',alpha=.55,label='Training data')
    ax.plot(grid,truth(grid),'--',color=RED,label='True mean')
    for r,colour in zip([runs[0],runs[2]],[BLUE,PURPLE]):
        ax.plot(grid,predict(r['final'],transform(grid)),color=colour,label=f"Width {r['width']}")
    ax.set(xlabel='x',ylabel='Response',ylim=(-1.7,1.7)); ax.legend(frameon=False,fontsize=11)
    save(fig,'nn-capacity.svg')
    fig, axs = plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for j,label in enumerate(['Training','Validation']):
        axs[0].plot(np.arange(1,1501),selected['history'][:,j],label=label,color=[BLUE,PURPLE][j])
    axs[0].axvline(selected['epoch'],color=RED,ls='--',label=f"Selected epoch {selected['epoch']}")
    axs[0].set(xlabel='Epoch',ylabel='MSE',yscale='log'); axs[0].legend(frameon=False,fontsize=10)
    fitplot(axs[1],selected['best'],'Selected checkpoint')
    axs[1].plot(grid,predict(selected['final'],transform(grid)),color=PURPLE,alpha=.7,label='Final epoch')
    axs[1].legend(frameon=False,fontsize=10)
    save(fig,'nn-selection.svg')
    table = ['<!-- Generated by nn_experiments.py; do not edit the numerical results by hand. -->',
             '| Width | L2 penalty | Selected epoch | Training MSE | Validation MSE |',
             '|--:|--:|--:|--:|--:|']
    for r in runs:
        tr = r['history'][r['epoch']-1,0]
        table.append(f"| {r['width']} | {r['penalty']:g} | {r['epoch']} | {tr:.4f} | {r['val']:.4f} |")
    table.extend(['',f"**Selected:** width {selected['width']}, $\\lambda={selected['penalty']:g}$; **test MSE:** {test_mse:.4f}.",
                  '', 'The noise variance is $0.04$; finite-sample test MSE can lie above or below this value.', ''])
    (BASE/'codes/nn_results.qmd').write_text('\n'.join(table))
    return dict(candidates=[dict(width=r['width'],penalty=r['penalty'],epoch=r['epoch'],validation_mse=r['val']) for r in runs],
                selected_width=selected['width'],selected_penalty=selected['penalty'],test_mse=test_mse)


def diagrams():
    from nn_network_diagrams import learning_graph
    learning_graph()
    fig, axs = plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    x = np.linspace(0,1,2001)
    for ax,n in zip(axs,[2,4,12]):
        knots=np.linspace(0,1,n+1)
        f=lambda t:np.sin(2*np.pi*t)*.7+t
        g=np.interp(x,knots,f(knots))
        ax.plot(x,f(x),color=RED,ls='--',label='Target')
        ax.plot(x,g,color=BLUE,label='Interpolant');ax.scatter(knots,f(knots),s=15,color=BLUE)
        ax.set(title=f'{n} intervals; max error {np.max(abs(f(x)-g)):.3f}',xlabel='x')
    axs[0].legend(frameon=False,fontsize=10)
    save(fig,'nn-approximation.svg')
    fig,axs=plt.subplots(1,3,figsize=(12,3.2),layout='constrained')
    T=lambda t:2*np.maximum(t,0)-4*np.maximum(t-.5,0)+2*np.maximum(t-1,0)
    y=x.copy()
    for depth,ax in enumerate(axs,1):
        y=T(y)
        ax.plot(x,y,color=[BLUE,PURPLE,RED][depth-1])
        ax.set(title=f'{depth} composition(s): {2**depth} pieces',xlabel='x',ylim=(-.05,1.05))
    save(fig,'nn-depth.svg')


if __name__=='__main__':
    error=check_gradients()
    results=regression()
    diagrams()
    results['maximum_gradient_check_error']=error
    (BASE/'codes/nn_experiment_results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
