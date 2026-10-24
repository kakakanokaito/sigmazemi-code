# 第3回の追加実験（シグマゼミ、2026-10-24）：kitten の近所、矢印の長さ、cos の手計算、いろいろな足し算・引き算。使い方：python3 more_probes.py <glove-wiki-gigaword-100.gz>
import gzip,sys,numpy as np
W=[];V=[]
with gzip.open(sys.argv[1],"rt",encoding="utf-8") as f:
    first=f.readline().split()
    if len(first)!=2: W.append(first[0]);V.append(np.array(first[1:],dtype=np.float32))
    for l in f:
        p=l.rstrip().split(" ");W.append(p[0]);V.append(np.array(p[1:],dtype=np.float32))
V=np.vstack(V);I={w:i for i,w in enumerate(W)};U=V/np.linalg.norm(V,axis=1,keepdims=True)
v=lambda w:V[I[w]]
def nn(q,k=5,ex=()):
    s=U@(q/np.linalg.norm(q));o=[]
    for i in np.argsort(-s):
        if W[i] in ex:continue
        o.append(f"{W[i]} {s[i]:.3f}")
        if len(o)==k:break
    return ", ".join(o)
print("|king|",np.linalg.norm(v("king")),"|woman-man|",np.linalg.norm(v("woman")-v("man")))
print("kitten近所:",nn(v("kitten"),6,("kitten",)))
print("cat近所:",nn(v("cat"),6,("cat",)))
print("a=(3,4) b=(4,3):",(3*4+4*3)/25)
qs=[("sushi","japan","italy"),("tokyo","japan","france"),("kagoshima","japan","italy"),("student","study","sleep"),("homework","school","home"),("monday","work","play"),("coffee","morning","night"),("einstein","physics","music"),("math","boring","fun"),("genius","smart","lazy"),("strawberry","red","green"),("teacher","school","hospital"),("pizza","italy","japan"),("beer","germany","japan"),("ninja","japan","america"),("samurai","japan","england")]
for a,b,c in qs:
    if all(w in I for w in (a,b,c)):
        print(f"{a}-{b}+{c}:",nn(v(a)-v(b)+v(c),5,(a,b,c)))
    else: print("missing",a,b,c,[w for w in (a,b,c) if w not in I])
for w in ["zundamon","zunda","edamame","mochi","kagoshima","sigma","natsume","soseki"]:
    print(w, w in I, nn(v(w),4,(w,)) if w in I else "")
