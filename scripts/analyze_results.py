"""Regenerate report figures/tables from saved logs; no training or checkpoints needed."""
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report_analysis';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def readlog(p):return pd.read_json(p,lines=True)
models=['rnn','gru','lstm','lstm_512','gpt_textbook','gpt_modern']
labels=['RNN','GRU','LSTM-256','LSTM-512','Textbook GPT','Modern GPT']
metrics={};summ=[]
for model in models:
 ms=[json.loads((ROOT/'runs/architecture'/f'{model}_seed{s}'/'metrics.json').read_text()) for s in [42,1337]]
 metrics[model]=ms
for model,label in zip(models,labels):
 ms=metrics[model];v=np.array([m['val_bpb'] for m in ms]);base=np.array([m['val_bpb'] for m in metrics['lstm']])
 summ.append({'model':model,'label':label,'params':ms[0]['params'],'mean_bpb':v.mean(),'std_bpb':v.std(ddof=1),'mean_train_sec':np.mean([m['train_time_sec'] for m in ms]),'delta_vs_lstm':(v-base).mean(),'paired_p_unadjusted':ttest_rel(v,base).pvalue if model!='lstm' else np.nan,'approx_PFLOPs_per_run':6*ms[0]['params']*ms[0]['tokens_seen']/1e15})
pd.DataFrame(summ).to_csv(OUT/'architecture_statistics.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4.0),layout='constrained')
for model,label in zip(models,labels):
 ds=[readlog(ROOT/'runs/architecture'/f'{model}_seed{s}'/'eval_log.jsonl') for s in [42,1337]]
 y=np.array([d.bpb for d in ds]);x=ds[0].train_tokens/1e6
 ax.plot(x,y.mean(0),label=label);ax.fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),alpha=.13)
ax.set(xlabel='Training tokens (millions)',ylabel='Validation bits/byte');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2);fig.savefig(OUT/'architecture_curves.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.6),layout='constrained')
for name,col in [('lstm','#2378a3'),('lstm_512','#b55b28')]:
 ds=[readlog(ROOT/'runs/architecture'/f'{name}_seed{s}'/'eval_log.jsonl') for s in [42,1337]]
 ts=[readlog(ROOT/'runs/architecture'/f'{name}_seed{s}'/'train_log.jsonl') for s in [42,1337]]
 ax.plot(ds[0].train_tokens/1e6,np.mean([d.loss for d in ds],axis=0),color=col,label=name+' validation')
 ax.plot(ts[0].step*8192/1e6,np.mean([t.loss for t in ts],axis=0),color=col,ls='--',label=name+' training minibatch')
ax.set(xlabel='Training tokens (millions)',ylabel='Loss (nats/token)',ylim=(.95,2.05));ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.savefig(OUT/'lstm_overfitting.png',dpi=180);plt.close(fig)
ctx=[];paired=[]
fig,ax=plt.subplots(figsize=(8,3.8),layout='constrained')
for name,label in [('lstm','LSTM (hidden size 256)'),('gpt_modern','Modern Transformer')]:
 vals={}
 for T in [64,128,256,512]:
  ms=[]
  for s in [42,1337]:
   path=ROOT/'runs'/('architecture' if T==256 else 'context')/(f'{name}_seed{s}' if T==256 else f'{name}_ctx{T}_seed{s}')/'metrics.json'
   ms.append(json.loads(path.read_text()))
  v=np.array([m['val_bpb'] for m in ms]);vals[T]=v
  ctx.append({'model':name,'context':T,'seed42':v[0],'seed1337':v[1],'mean_bpb':v.mean(),'std_bpb':v.std(ddof=1),'val_tokens':ms[0]['val_tokens']})
 ax.errorbar(list(vals),[v.mean() for v in vals.values()],yerr=[v.std(ddof=1) for v in vals.values()],marker='o',capsize=4,label=label)
 for a,b in [(64,128),(128,256),(256,512),(64,512)]:
  d=vals[a]-vals[b];paired.append({'model':name,'from':a,'to':b,'gain_seed42':d[0],'gain_seed1337':d[1],'mean_gain':d.mean(),'paired_p_unadjusted':ttest_rel(vals[a],vals[b]).pvalue})
ax.set(xlabel='Training and evaluation context (characters)',ylabel='Validation bits/byte',xticks=[64,128,256,512]);ax.legend();ax.grid(alpha=.2);fig.savefig(OUT/'context.png',dpi=180);plt.close(fig)
pd.DataFrame(ctx).to_csv(OUT/'context_statistics.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'context_paired_tests.csv',index=False)
texts=json.loads((ROOT/'attention/texts.json').read_text())
fig,axes=plt.subplots(2,3,figsize=(12,8),layout='constrained')
for j,(kind,text) in enumerate(texts.items()):
 maps=np.load(ROOT/'attention'/f'{kind}_weights.npz')['attention']
 assert np.allclose(maps.sum(-1),1,atol=1e-6) and np.max(np.triu(maps,1))==0
 for i,(layer,head) in enumerate([(0,0),(2,0)]):
  ax=axes[i,j];im=ax.imshow(maps[layer,head],vmin=0,vmax=1,cmap='Blues');ax.set_title(f'{kind}: layer {layer}, head {head}')
  ticks=np.arange(0,len(text),5);ax.set_xticks(ticks);ax.set_yticks(ticks);ax.set_xlabel('Key position');ax.set_ylabel('Query position')
fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Attention weight');fig.savefig(OUT/'selected_attention_heads.png',dpi=180);plt.close(fig)
m=metrics['gpt_modern'][1];L=m['val_loss'];N=m['val_tokens'];B=m['val_bytes'];calc={'run':'gpt_modern_seed1337','val_loss':L,'val_tokens':N,'val_bytes':B,'reconstructed_total_nll_nats':L*N,'ppl_calculated':math.exp(L),'ppl_logged':m['val_ppl'],'bits_per_token':L/math.log(2),'bpb_calculated':L/math.log(2)*N/B,'bpb_logged':m['val_bpb']}
assert math.isclose(calc['ppl_calculated'],m['val_ppl'],rel_tol=1e-12);assert math.isclose(calc['bpb_calculated'],m['val_bpb'],rel_tol=1e-12)
(OUT/'metric_calculations.json').write_text(json.dumps(calc,indent=2))
print(json.dumps(calc,indent=2));print('Regenerated all report figures and tables.')
