import mesa, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class CitizenAgent(mesa.Agent):
    def __init__(self, model, alpha, eps=None):
        super().__init__(model)
        self.opinion = self.model.random.random()
        self.alpha=alpha; self.eps=eps
    def step(self):
        nb = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
        if self.eps is not None:
            nb=[n for n in nb if abs(n.opinion-self.opinion)<self.eps]
        if nb:
            m=np.mean([n.opinion for n in nb])
            self.opinion += self.alpha*(m-self.opinion)

class M(mesa.Model):
    def __init__(self,w=10,h=10,alpha=.5,eps=None,seed=42):
        super().__init__(seed=seed)
        self.grid=mesa.space.MultiGrid(w,h,torus=True)
        for x in range(w):
            for y in range(h):
                self.grid.place_agent(CitizenAgent(self,alpha,eps),(x,y))
        self.hist=[]
    def rec(self):
        o=[a.opinion for a in self.agents]; self.hist.append((np.mean(o),np.var(o)))
    def step(self):
        self.rec(); self.agents.shuffle_do("step")

def run(alpha,eps=None,steps=50,seed=42):
    m=M(alpha=alpha,eps=eps,seed=seed)
    for _ in range(steps): m.step()
    m.rec()
    return m

# convergence speed
print("alpha | var@t=10 | var@t=50 | 1st tick var<1e-3 | mean@50")
fig,ax=plt.subplots(figsize=(9,5))
for a in [0.05,0.2,0.5,0.8,1.0]:
    m=run(a); v=np.array([h[1] for h in m.hist]); mu=m.hist[-1][0]
    t=next((i for i,x in enumerate(v) if x<1e-3),None)
    print(a, f"{v[10]:.2e}", f"{v[-1]:.2e}", t, f"{mu:.3f}")
    ax.semilogy(v,label=f"α={a}")
ax.set_xlabel("Ticks"); ax.set_ylabel("Variance (log)"); ax.set_title("Vitesse de convergence selon α"); ax.legend()
fig.savefig("captures/03_convergence_selon_alpha.png",dpi=130,bbox_inches="tight")

# bounded confidence
fig,axs=plt.subplots(1,3,figsize=(15,4.5),sharey=True)
for ax,eps in zip(axs,[None,0.3,0.1]):
    m=M(alpha=.5,eps=eps,seed=42)
    tr=[]
    for _ in range(100):
        tr.append([a.opinion for a in sorted(m.agents,key=lambda a:a.unique_id)]); m.step()
    tr=np.array(tr)
    ax.plot(tr[:,:30],alpha=.6); ax.set_title("sans seuil" if eps is None else f"ε = {eps}"); ax.set_xlabel("Ticks")
    final=np.array(sorted(a.opinion for a in m.agents))
    clusters=1+int(np.sum(np.diff(final)>0.05))
    print("eps",eps,"clusters",clusters,"var",np.var(final),"min/max",final.min(),final.max())
axs[0].set_ylabel("Opinion")
fig.suptitle("Biais de confirmation (confiance bornée)")
fig.savefig("captures/04_biais_confirmation.png",dpi=130,bbox_inches="tight")
