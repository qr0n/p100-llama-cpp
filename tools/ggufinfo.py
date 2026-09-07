import struct, sys, collections

TT={}
for line in open(sys.argv[2] if len(sys.argv)>2 else '/dev/null'):
    p=line.split()
    if len(p)==4: TT[int(p[0])]=(p[1],int(p[2]),int(p[3]))

f=open(sys.argv[1],'rb')
def rd(n): return f.read(n)
def u32(): return struct.unpack('<I',rd(4))[0]
def u64(): return struct.unpack('<Q',rd(8))[0]
def i64(): return struct.unpack('<q',rd(8))[0]
def s():
    n=u64(); return rd(n).decode('utf-8','replace')

assert rd(4)==b'GGUF'
ver=u32(); n_tensors=u64(); n_kv=u64()

# value readers by gguf metadata type
def rdval(t):
    if t==0: return struct.unpack('<B',rd(1))[0]
    if t==1: return struct.unpack('<b',rd(1))[0]
    if t==2: return struct.unpack('<H',rd(2))[0]
    if t==3: return struct.unpack('<h',rd(2))[0]
    if t==4: return u32()
    if t==5: return struct.unpack('<i',rd(4))[0]
    if t==6: return struct.unpack('<f',rd(4))[0]
    if t==7: return struct.unpack('<?',rd(1))[0]
    if t==8: return s()
    if t==9:
        et=u32(); n=u64(); return [rdval(et) for _ in range(n)]
    if t==10: return u64()
    if t==11: return i64()
    if t==12: return struct.unpack('<d',rd(8))[0]
    raise ValueError('bad kv type %d'%t)

kv={}
for _ in range(n_kv):
    k=s(); t=u32(); v=rdval(t)
    kv[k]=v

tensors=[]
for _ in range(n_tensors):
    name=s(); nd=u32(); dims=[u64() for _ in range(nd)]
    ttype=u32(); off=u64()
    ne=1
    for d in dims: ne*=d
    tn,bs,ts=TT[ttype]
    nbytes=ne//bs*ts
    tensors.append((name,dims,tn,nbytes))

print('gguf v%d  tensors=%d  kv=%d'%(ver,n_tensors,n_kv))
for k in ('general.architecture','llama.block_count','llama.embedding_length',
          'llama.attention.head_count','llama.attention.head_count_kv',
          'llama.feed_forward_length','llama.context_length','llama.rope.dimension_count'):
    if k in kv: print('  %-38s %s'%(k,kv[k]))

tot=sum(t[3] for t in tensors)
emb=sum(t[3] for t in tensors if 'token_embd' in t[0])
outw=sum(t[3] for t in tensors if t[0].startswith('output.'))
bytype=collections.Counter()
for t in tensors: bytype[t[2]]+=t[3]

G=1024**3
print()
print('total tensor bytes    %15d  %8.4f GiB'%(tot,tot/G))
print('token_embd            %15d  %8.4f GiB   (decode reads ONE row)'%(emb,emb/G))
print('output head           %15d  %8.4f GiB   (read in full each token)'%(outw,outw/G))
print('DECODE READ / TOKEN   %15d  %8.4f GiB'%(tot-emb,(tot-emb)/G))
print()
print('by quant type:')
for k,v in bytype.most_common(): print('   %-8s %15d  %8.4f GiB  (%5.2f%%)'%(k,v,v/G,100*v/tot))
