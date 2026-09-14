import csv,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from concurrent.futures import ThreadPoolExecutor
out=Path(r'C:\Users\T3USER\research\PatchmatchNet\depth_audit_20260913');out.mkdir(exist_ok=True)
base=Path(r'C:\Users\T3USER\Desktop\Data Preprocessing')
summary={}
for label in ['20cm','35cm']:
 root=base/f'20260913_{label}_output'
 pairs=list(csv.DictReader((root/'pairs.csv').open(encoding='utf-8-sig')))
 cal=json.loads((root/'calibration.json').read_text(encoding='utf8'))
 result={'pairs':len(pairs),'calibration':cal,'delta_ms_percentiles':np.percentile([abs(float(r['delta_ms'])) for r in pairs],[0,50,95,100]).tolist(),'scales':sorted(set(r['depth_scale_m_per_unit'] for r in pairs))}
 for folder in ['depth_raw','depth_aligned']:
  files=sorted((root/folder).glob('*'))
  def scan(p):
   try:
    a=np.array(Image.open(p));v=a>0;h,w=a.shape;q=a[v]; c=v[h//4:3*h//4,w//4:3*w//4]
    return {'file':p.name,'shape':str(a.shape),'dtype':str(a.dtype),'valid':float(v.mean()),'center_valid':float(c.mean()),'min':int(q.min()) if q.size else 0,'max':int(q.max()) if q.size else 0,'median':float(np.median(q)) if q.size else 0,'saturated':int((a==65535).sum())}
   except Exception as e:return {'file':p.name,'error':str(e)}
  with ThreadPoolExecutor(max_workers=8) as pool: rows=list(pool.map(scan,files))
  (out/f'{label}_{folder}.json').write_text(json.dumps(rows),encoding='utf8')
  good=[r for r in rows if 'error' not in r]
  result[folder]={'count':len(files),'errors':[r for r in rows if 'error'in r],'shapes':sorted(set(r['shape'] for r in good)),'dtypes':sorted(set(r['dtype'] for r in good)),'valid_mean':float(np.mean([r['valid'] for r in good])),'valid_min_median_max':np.percentile([r['valid'] for r in good],[0,50,100]).tolist(),'center_valid_mean':float(np.mean([r['center_valid'] for r in good])),'median_depth_mm_min_median_max':np.percentile([r['median'] for r in good],[0,50,100]).tolist(),'absolute_range':[min(r['min'] for r in good),max(r['max'] for r in good)],'saturated_pixels':sum(r['saturated'] for r in good)}
  print(label,folder,result[folder],flush=True)
 result['missing_pairs']=sum(not(root/r[k]).is_file() for r in pairs for k in ['rgb_file','depth_raw_file','depth_aligned_file','depth_preview_file'])
 summary[label]=result
 # Samples across recording. Metric color scale 0-1000 mm, zero black.
 canvas=Image.new('RGB',(960,4*260),'white');draw=ImageDraw.Draw(canvas)
 for j,idx in enumerate(np.linspace(0,len(pairs)-1,4,dtype=int)):
  row=pairs[idx]
  for col,key in enumerate(['rgb_file','depth_raw_file','depth_aligned_file']):
   im=Image.open(root/row[key])
   if col:
    a=np.array(im);t=np.clip(a/1000,0,1);rgb=np.stack([255*t,255*(1-abs(2*t-1)),255*(1-t)],axis=-1).astype('uint8');rgb[a==0]=0;im=Image.fromarray(rgb)
   im.thumbnail((320,230));canvas.paste(im,(col*320,j*260+25));draw.text((col*320+3,j*260+5),f'{label} {idx:06d} {key}',fill='black')
 canvas.save(out/f'{label}_samples.jpg')
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print('DONE',out)
