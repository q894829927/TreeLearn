"""Re-render cached D0 votes on common metre axes; no GPU/model/input changes."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--diagnosis',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for folder in sorted(Path(args.diagnosis).glob('*/calibrated_eval_inner.npz')):
        case=folder.parent
        with np.load(folder) as data:cal=dict(data)
        with np.load(case/'batch_bn_inner.npz') as data:batch=data['offsets']
        labels=cal['labels'];rng=np.random.default_rng(42)
        ids=np.unique(labels[labels>=0])
        ix=np.concatenate([rng.choice(points,min(len(points),256),replace=False)
                           for label in ids if len(points:=np.flatnonzero(labels==label))])[:10000]
        # Consecutive GT IDs avoid collisions caused by label modulo a palette.
        categories=np.searchsorted(ids,labels[ix]);cmap=plt.get_cmap('tab20',len(ids))
        colors=cmap(categories);xy=cal['coords'][ix,:2]
        points=[xy,xy+cal['target'][ix,:2],xy+cal['offsets'][ix,:2],xy+batch[ix,:2]]
        combined=np.concatenate(points);low=combined.min(0);high=combined.max(0);margin=.04*np.maximum(high-low,1.)
        fig,axes=plt.subplots(1,4,figsize=(16,4.5),sharex=True,sharey=True)
        for ax,title,values in zip(axes,['input XY','GT votes','calibrated BN votes','batch BN votes'],points):
            ax.scatter(values[:,0],values[:,1],c=colors,s=1)
            ax.set_title(title);ax.set_aspect('equal');ax.set_xlim(low[0]-margin[0],high[0]+margin[0]);ax.set_ylim(low[1]-margin[1],high[1]+margin[1]);ax.set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        handles=[plt.Line2D([0],[0],marker='o',color=cmap(i),linestyle='',label=str(label),markersize=4) for i,label in enumerate(ids)]
        fig.legend(handles=handles,loc='lower center',ncol=min(len(ids),12),title='GT instance ID; 0 = known background')
        fig.suptitle(case.name+'; common axes; output-only display sample; full input evaluated')
        fig.tight_layout(rect=(0,.16,1,.94));path=out/(case.name+'_votes_shared_axes.png')
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':main()
