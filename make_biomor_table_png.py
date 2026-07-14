"""Render the completed bioMoR efficiency ladder in the MoR-paper table format as a PNG.
All numbers pulled from results_arch13/ (single-cell) + results_pw13/ (multi-omics) and
NLL from paper Table 10.  '-' = variant genuinely not run (verified by full results scan)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ----- leaf columns --------------------------------------------------------
cols = ["Type","KV","Share","$N_R$","Param","FLOPs","$N_{tok}$","mean",
        "Bar","Lun","Mur","Oes","Seg","Spl","Tce","Xin","Pro","BL","ST","Avg"]
groups = [("MoR",0,2),("Recursion",2,4),("Model",4,7),("NLL$\\downarrow$",7,8),
          ("Single-cell macro-F1$\\uparrow$",8,16),("Multi-omics",16,19),("",19,20)]
lead, right, X0, XMOD = 0.155, 0.997, 0.004, 0.078
n=len(cols)
xs=[lead+(right-lead)*(i+0.5)/n for i in range(n)]
def leafx(i): return xs[i]
def edge(i): return lead+(right-lead)*i/n

D="–"
# row = (model_label, [20 vals], style)   style: n / head(shade+bold) / RULE
rows = [
 ("Vanilla",      [D,D,D,"4","300K","1.00×",D,"1.30",
                   "62.5","70.4","67.8","55.8","58.1","48.4","48.9","65.9","54.5","40.0","53.0","56.8"], "n"),
 ("RULE",),
 ("Recursive$^\\dagger$",[D,D,"Cyc","2","75K","1.00×",D,D, *([D]*11), D], "n"),
 ("",             [D,D,"Cyc","3","75K","1.00×",D,D, *([D]*11), D], "n"),
 ("",             [D,D,"Cyc","4","75K","1.00×",D,"1.36",
                   "63.0","73.6","73.2","55.0","59.1","49.0","48.5","66.1","57.9","44.5","50.0","58.2"], "n"),
 ("RULE",),
 ("MoR (general)",["Expert",D,"Cyc","1","75K","1.00×",D,D,
                   "62.5","71.6","66.9","57.0","50.8","49.3","50.1","65.7","50.4","36.0","40.7","54.6"], "n"),
 ("",             ["Expert",D,"Cyc","2","75K","0.62×",D,D, *([D]*11), D], "n"),
 ("",             ["Expert",D,"Cyc","3","75K","0.62×",D,D, *([D]*11), D], "n"),
 ("",             ["Expert",D,"Cyc","4","75K","0.62×",D,"1.29",
                   "58.6","71.4","68.9","55.0","56.9","49.0","49.2","68.2","53.8","42.2","43.9","56.1"], "n"),
 ("RULE",),
 ("bioMoR (ours)",["Expert",D,"Cyc","4","75K","0.62×",D,"1.31",
                   "79.1","79.1","75.3","69.6","71.4","58.2","68.9","64.7","78.2","40.9","52.2","67.1"], "head"),
 ("RULE",),
 ("MoR (general)",["Token",D,"Cyc","4","75K","0.56×",D,D,
                   "63.7","73.0","75.0","56.5","62.6","48.8","49.6","61.5","76.3","42.8","46.9","59.7"], "n"),
 ("RULE",),
 ("",             ["Expert","Cache","Cyc","4","75K",D,D,D, *([D]*11), D], "n"),
 ("",             ["Expert",D,"Shr","4","75K",D,D,D, *([D]*11), D], "n"),
]

# ----- figure --------------------------------------------------------------
n_body=sum(1 for r in rows if r[0]!="RULE")
n_rule=sum(1 for r in rows if r[0]=="RULE")
fig,ax=plt.subplots(figsize=(15.2,0.95+0.46*n_body+0.12*n_rule),dpi=200)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

top=0.95; head_h=0.055; row_h=0.058; fs=10.2; hfs=11
y_grp=top; y_leaf=top-head_h
def line(x0,x1,yy,lw=1.0): ax.plot([x0,x1],[yy,yy],color="black",lw=lw,solid_capstyle="butt")

line(X0,right,top+0.030,1.6)                       # top rule
for name,a,b in groups:
    if not name: continue
    cx=(edge(a)+edge(b))/2
    ax.text(cx,y_grp,name,ha="center",va="center",fontsize=hfs,fontweight="bold")
    if b-a>1: line(edge(a)+0.004,edge(b)-0.004,y_grp-0.026,0.9)
ax.text(XMOD,y_leaf,"Models",ha="center",va="center",fontsize=hfs,fontweight="bold")
for i,lab in enumerate(cols):
    ax.text(leafx(i),y_leaf,lab,ha="center",va="center",fontsize=hfs-0.6,fontweight="bold")
line(X0,right,y_leaf-0.028,1.2)                    # rule under headers

y=y_leaf-0.028-0.006
for r in rows:
    if r[0]=="RULE":
        y-=0.010; line(X0,right,y,0.7); y-=0.006; continue
    label,vals,style=r
    yy=y-row_h/2
    if style=="head":
        ax.add_patch(Rectangle((X0,yy-row_h/2+0.004),right-X0,row_h-0.004,
                     facecolor="#d9e6f2",edgecolor="none",zorder=0))
    bold=style=="head"
    if label:
        lb="bold" if (label.startswith("bioMoR") or label.startswith("MoR")) else "normal"
        ax.text(X0+0.006,yy,label,ha="left",va="center",fontsize=fs,fontweight=lb)
    for i,v in enumerate(vals):
        ax.text(leafx(i),yy,v,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
    y-=row_h

line(X0,right,y+0.004,1.6)                          # bottom rule
cap_y=y-0.02
ax.text(X0,cap_y,
 "Completed bioMoR efficiency ladder in the MoR-table format. General MoR (biology-free, expert & token) brackets the bioMoR headline. "
 "Columns: routing Type / KV-reuse / weight-Share strategy / recursion depth $N_R$; transformer-stack Param, relative recursion FLOPs, "
 "pretraining tokens $N_{tok}$ (n/a); mean raw NLL over 11 datasets (Table 10); per-dataset macro-F1 (8 single-cell + 3 multi-omics) and Avg.",
 ha="left",va="top",fontsize=7.8,style="italic")
ax.text(X0,cap_y-0.055,
 "Shaded row = headline (expert-choice MoR + learned gene-graph routing, +11.0 F1 over general MoR). '–' = variant not run "
 "(full results scan: only $N_R\\in\\{1,4\\}$ exist; KV=Cache step-cache and Share=Shr strategy were never run; Token-choice NLL not logged).",
 ha="left",va="top",fontsize=7.8,style="italic")

plt.subplots_adjust(left=0.004,right=0.997,top=0.995,bottom=0.01)
out="/work/mech-ai-scratch/tirtho/RecusrsiveQFormer/biomor_ladder_table.png"
fig.savefig(out,dpi=200,bbox_inches="tight",facecolor="white")
print("wrote",out)
