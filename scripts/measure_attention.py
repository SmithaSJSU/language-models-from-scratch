"""Recompute selected-head measurements from the saved attention arrays."""
from pathlib import Path
import json,re
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1]
rows=[]
for kind,text in json.loads((root/'attention/texts.json').read_text()).items():
 a=np.load(root/'attention'/f'{kind}_weights.npz')['attention']
 t=np.arange(1,len(text))
 rows.append(dict(text=kind,layer=0,head=0,target='previous_character',n=len(t),attention=a[0,0,t,t-1].mean(),previous_attention=a[0,0,t,t-1].mean(),uniform=np.mean(1/(t+1))))
 pairs=[(t,m.start()) for m in re.finditer('[A-Za-z]+',text) for t in range(m.start()+2,m.end())]
 t,s=map(np.array,zip(*pairs))
 rows.append(dict(text=kind,layer=2,head=0,target='word_start_distinct',n=len(t),attention=a[2,0,t,s].mean(),previous_attention=a[2,0,t,t-1].mean(),uniform=np.mean(1/(t+1))))
df=pd.DataFrame(rows);df.to_csv(root/'report_analysis/selected_head_measurements.csv',index=False)
print(df.to_string(index=False));print(df.groupby('target')[['attention','previous_attention','uniform']].mean())
