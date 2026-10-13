from pathlib import Path
import sys,os,json,csv
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path.insert(0,str(ROOT/'experiments/2026-10-12_pressure_fixed_k/.deps'))
os.environ['MPLCONFIGDIR']=str(H/'.mplconfig')
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':10})
KB=1.380649e-23;HP=6.62607015e-34;EV=1.602176634e-19;R=8.314462618
checks=[]
def matrix(tc,p,step,closure):
 T=tc+273.15;C=-2.1661e-8*tc+1.68935e-5
 wd,wm=(1,2) if closure=='A' else (2,1)
 # State fractions: OH,Tads,D,Dwater,B,Bwater,M,Mwater; last state accumulated Al/C.
 A=np.zeros((9,9))
 def link(i,targets,k):
  A[i,i]-=k
  for j,f in targets:A[j,i]+=k*f
 def rate(e):return KB*T/HP*np.exp(-e*EV/(KB*T))
 ft=p*133.322368/np.sqrt(2*np.pi*.07209*R*T)
 fw=p*133.322368/np.sqrt(2*np.pi*.01801528*R*T)
 if step=='TMA':link(0,[(1,1)],1.5*.004*ft/C)
 link(1,[(0,1)],rate(.61))
 link(1,[(2,wd/3),(6,wm/3)],rate(.52));A[8,1]=rate(.52)/1.5
 for parent,adduct,w,er,ed,targets in [(2,3,wd,.70,.57,[(4,1/wd),(0,(wd-1)/wd)]),(4,5,1,.91,.74,[(0,1)]),(6,7,wm,.70,.57,[(0,1)])]:
  if step=='water':link(parent,[(adduct,1)],w*.014*fw/C)
  link(adduct,[(parent,1)],rate(ed));link(adduct,targets,rate(er))
 assert np.max(np.abs(A[:8,:8].sum(axis=0)))<1e-5
 return A
def calc(tc,p,pulse,closure):
 purge=float(np.interp(tc,[125,150,162,175,200,250,300],[30,20,20,15,10,8,5]))
 mats=[matrix(tc,p,'TMA',closure),matrix(tc,0,'purge',closure),matrix(tc,p,'water',closure)]
 ops=[expm(mats[0]*pulse/1000),expm(mats[1]*purge),expm(mats[2]*.1),expm(mats[1]*purge)]
 U=ops[3]@ops[2]@ops[1]@ops[0]
 y=np.zeros(9);y[0]=1;growth=[];err=0;mn=1
 for i in range(550 if tc==300 else 500):
  y[8]=0;y=U@y;growth.append(y[8]);err=max(err,abs(y[:8].sum()-1));mn=min(mn,y[:8].min())
 C=-2.1661e-8*tc+1.68935e-5;factor=C*.101961276/(2*3500)*1e10
 assert err<1e-5 and mn>-1e-8
 return dict(T_C=float(tc),P_Torr=p,pulse_ms=pulse,closure=closure,GPC_mean=float(np.mean(growth)*factor),GPC_last=float(growth[-1]*factor),cycle_change=float(abs(growth[-1]-growth[-2])*factor),balance_error=float(err),min_fraction=float(mn),OH_end=float(y[0]),Me_occupied_end=float(sum(y[2:8])))
data=list(csv.DictReader((H.parent/'2026-10-13_single_source/fig3_experiment.csv').open()))
temps=sorted(set(list(range(125,301,5))+[162]))
rows=[calc(t,p,pulse,c) for c in ['A','B'] for p in [.01,.03,.1,.3] for pulse in [25,60] for t in temps]
metrics=[]
for c in ['A','B']:
 for p in [.01,.03,.1,.3]:
  pred=[];obs=[]
  for d in data:
   rec=next(z for z in rows if z['T_C']==float(d['T_C']) and z['pulse_ms']==int(d['pulse_ms']) and z['P_Torr']==p and z['closure']==c)
   pred.append(rec['GPC_mean']);obs.append(float(d['GPC']))
  pred=np.array(pred);obs=np.array(obs)
  metrics.append(dict(closure=c,P_Torr=p,RMSE_A=float(np.sqrt(np.mean((pred-obs)**2))),MAPE_pct=float(np.mean(abs(pred/obs-1))*100)))
# Independent ODE comparison for a stiff half-cycle and closed-form two-state survival.
A=matrix(150,.03,'TMA','A');y=np.zeros(9);y[0]=1
s=solve_ivp(lambda t,y:A@y,[0,.025],y,method='Radau',rtol=1e-9,atol=1e-12)
err=float(max(abs(s.y[:,-1]-expm(A*.025)@y)));assert err<1e-7
# At zero incoming flux one adsorbate reacts or desorbs with known branching probability.
A=matrix(150,0,'purge','A');y=np.zeros(9);y[1]=1
z=expm(A*.001)@y;q=1/(1+np.exp(-(.61-.52)*EV/(KB*(150+273.15))))
ae=abs(z[8]-q/1.5);assert ae<1e-7
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for i,c in enumerate(['A','B']):
 for j,pulse in enumerate([25,60]):
  ax=axs[i,j]
  for p in [.01,.03,.1,.3]:
   rr=[x for x in rows if x['closure']==c and x['pulse_ms']==pulse and x['P_Torr']==p]
   ax.plot([x['T_C'] for x in rr],[x['GPC_mean'] for x in rr],label=f'가정 분압 {p:g} Torr')
  dd=[d for d in data if int(d['pulse_ms'])==pulse]
  ax.scatter([float(d['T_C']) for d in dd],[float(d['GPC']) for d in dd],color='black',marker='s',s=28,label='문헌 실험점',zorder=5)
  ax.set(title=f'자리 수지 {c} / TMA {pulse} ms',xlabel='온도 (C)',ylabel='Al 기준 당량 GPC (A/cycle)',ylim=(0,1.55));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Gakis 계수 기반 비교 모델: 분압과 자리 수지는 가정 / GPC 피팅 없음',fontsize=14)
fig.tight_layout();fig.savefig(H/'gpc_comparison.png',dpi=160)
for name,rr in [('results.csv',rows),('metrics.csv',metrics)]:
 with (H/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
checks={'ODE_expm_error':err,'analytic_branch_error':float(ae),'max_balance_error':max(x['balance_error'] for x in rows),'min_fraction':min(x['min_fraction'] for x in rows),'max_last_cycle_change_A':max(x['cycle_change'] for x in rows)}
(H/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps({'metrics':metrics,'checks':checks},indent=2))
