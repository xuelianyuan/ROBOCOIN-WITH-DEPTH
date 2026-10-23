let views=[];const methods=['metric','nested','refine'];
const labels={chest:'CHEST',head:'HEAD',left:'LEFT WRIST',right:'RIGHT WRIST',metric:'Metric',nested:'Nested',refine:'Refine'};
const state={meta:null,frame:0,pcdFrame:0,playing:false,loops:1,last:0,lastMedia:0,offsets:{}};
const $=s=>document.querySelector(s), clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

// ============================== 初始化与表格布局 ==============================
async function load(){const meta=await fetch('/api/meta').then(r=>r.json());state.meta=meta;if(meta.error){$('#status').textContent=`加载失败：${meta.error}`;return}if(!meta.ready){$('#status').textContent=meta.status||'正在分析深度视频...';setTimeout(load,700);return}init()}
function init(){
  views=state.meta.views?.length?state.meta.views:['head','left','right'];
  $('#title').textContent=state.meta.title;$('#status').textContent='全视频有效深度统计已完成';
  $('#table').style.gridTemplateColumns=`150px repeat(${views.length},minmax(330px,1fr))`;
  $('#table').innerHTML=headerRow()+rgbRow()+methods.map(method=>methodRow(method)).join('')+curveRow();
  document.querySelectorAll('.offset input').forEach(input=>input.oninput=()=>{state.offsets[input.dataset.id]=+input.value;input.nextElementSibling.textContent=signed(+input.value);syncRgb(true);renderMedia()});
  $('#timeline').max=Math.max(0,maxFrames()-1);$('#timeline').oninput=e=>seek(+e.target.value);$('#play').onclick=toggle;
  initPcdFrameSelector();renderPcds();
  render(true);state.playing=true;$('#play').textContent='Ⅱ';state.last=performance.now();syncRgb(true);requestAnimationFrame(tick);
}
function headerRow(){return `<div class="corner">RESULT</div>${views.map(v=>`<div class="column-heading">${labels[v]}</div>`).join('')}`}
function rgbRow(){return `<div class="row-heading"><b>Raw RGB</b><span>同步原始画面</span></div>${views.map(v=>{const info=state.meta.rgb[v];return `<div class="rgb-cell">${info?`<video id="rgb-${v}" src="${info.url}" muted playsinline></video>`:'<div class="empty">无 RGB 视频</div>'}</div>`}).join('')}`}
function methodRow(method){return `<div class="row-heading"><b>${labels[method]}</b><span>Depth + 颜色直方图 + PCD</span></div>${views.map(v=>methodCell(v,method)).join('')}`}
function methodCell(view,method){const id=`${view}.${method}`,g=state.meta.groups[id]||{};state.offsets[id]=0;if(!g.available)return `<article class="method-cell"><div class="empty">${g.error||'无数据'}</div></article>`;const pcd=g.pcd_video?`<video src="${g.pcd_video_url}" muted playsinline preload="auto" aria-label="${id} point cloud"></video>`:`<div class="empty">${g.pcd?'读取点云':'未配置 PCD'}</div>`;return `<article class="method-cell" data-id="${id}"><div class="cell-meta"><b>${labels[method]}</b><span>方法 ${fmt(g.min)}–${fmt(g.max)} · 归一 ${fmt(g.normalize_min)}–${fmt(g.normalize_max)}</span></div><div class="media-pair"><figure><figcaption>DEPTH</figcaption><div class="viewport depth-view"><video src="${g.url}" muted playsinline preload="auto" aria-label="${id} depth"></video></div></figure><figure><figcaption>颜色直方图</figcaption><div class="viewport histogram-view"><canvas width="240" height="155"></canvas></div></figure><figure><figcaption>${g.pcd_video?'PCD VIDEO':'PCD'}</figcaption><div class="viewport pcd-view">${pcd}</div></figure></div><label class="offset">帧偏移<input data-id="${id}" type="range" min="-30" max="30" value="0"><span>0</span></label></article>`}
function curveRow(){return `<div class="row-heading"><b>均值曲线</b><span>纵轴：深度值 · 横轴：帧号</span></div>${views.map(v=>`<div class="curve"><div class="curve-legend"><span>● Metric</span><span>● Nested</span><span>● Refine</span></div><canvas id="curve-${v}" width="560" height="170"></canvas></div>`).join('')}`}
function fmt(n){return Number.isFinite(n)?n.toFixed(1):'-'}function signed(n){return n>0?`+${n}`:`${n}`}
function maxFrames(){return Math.max(1,...Object.values(state.meta.groups).filter(x=>x.available).map(x=>x.frames))}
function groupFrame(group,offset=0){const globalFrame=clamp(state.frame+offset,0,maxFrames()-1);return Math.round(globalFrame/Math.max(1,maxFrames()-1)*Math.max(0,group.frames-1))}
function sampledPcdFrames(){return [...new Set(Object.values(state.meta.groups).flatMap(group=>group.pcd_frames||[]))].sort((a,b)=>a-b)}
function initPcdFrameSelector(){const input=$('#pcd-frame'),frames=sampledPcdFrames(),hasVideo=Object.values(state.meta.groups).some(group=>group.pcd_video),initial=frames[0]??0;if(!frames.length&&hasVideo){document.querySelector('.pcd-frame-control').style.display='none';return}state.pcdFrame=clamp(initial,0,maxFrames()-1);input.min=0;input.max=maxFrames()-1;input.value=state.pcdFrame;$('#pcd-frame-options').innerHTML=frames.map(frame=>`<option value="${frame}"></option>`).join('');$('#pcd-frame-hint').textContent=frames.length?`已抽取 ${frames.length} 帧：${frames.slice(0,20).join('、')}${frames.length>20?'…':''}`:'未发现抽帧提示，可输入任意帧';input.onchange=()=>selectPcdFrame(input.value);input.onkeydown=event=>{if(event.key==='Enter'){event.preventDefault();input.blur()}}}
function selectPcdFrame(value){const parsed=Number(value);if(!Number.isFinite(parsed)){$('#pcd-frame').value=state.pcdFrame;return}state.pcdFrame=Math.round(clamp(parsed,0,maxFrames()-1));$('#pcd-frame').value=state.pcdFrame;renderPcds()}

// ============================== 播放与统一时间轴 ==============================
function toggle(){state.playing=!state.playing;$('#play').textContent=state.playing?'Ⅱ':'▶';if(state.playing&&state.frame>=maxFrames()-1&&state.loops>=3){state.frame=0;state.loops=1}state.last=performance.now();syncRgb(true)}
function seek(frame){state.frame=clamp(frame,0,maxFrames()-1);state.last=performance.now();syncRgb(true);render(true)}
function fps(){return Object.values(state.meta.groups).find(x=>x.available)?.fps||30}
function masterVideo(){return views.map(view=>$(`#rgb-${view}`)).find(Boolean)}
function syncRgb(force=false){
  const master=masterVideo();
  function sync(video,time,rate=1,atBoundary=false){
    if(!video)return;
    if(video.readyState===0){video.onloadedmetadata=()=>syncRgb(true);return}
    const sampled=rate<.0625;
    if(force||(video!==master&&!video.seeking&&Math.abs(video.currentTime-time)>(sampled?.001:.25)))video.currentTime=time;
    video.playbackRate=sampled?1:rate;
    if(state.playing&&!atBoundary&&!sampled){if(video.paused)video.play().catch(()=>{})}else video.pause();
  }
  views.forEach(view=>sync($(`#rgb-${view}`),state.frame/fps()));
  document.querySelectorAll('.method-cell[data-id]').forEach(cell=>{
    const id=cell.dataset.id,g=state.meta.groups[id],offset=state.offsets[id];
    const rate=(g.frames-1)/Math.max(1,maxFrames()-1)*fps()/g.fps;
    sync(cell.querySelector('.depth-view video'),groupFrame(g,offset)/g.fps,rate,state.frame+offset<0||state.frame+offset>=maxFrames()-1);
    sync(cell.querySelector('.pcd-view video'),groupFrame(g,offset)/g.fps,rate,state.frame+offset<0||state.frame+offset>=maxFrames()-1);
  });
}
function tick(now){
  if(state.playing){
    const master=masterVideo();
    if(master){if(!master.seeking)state.frame=clamp(Math.floor(master.currentTime*fps()),0,maxFrames()-1)}
    else{const steps=Math.floor((now-state.last)*fps()/1000);state.frame+=steps;state.last+=steps*1000/fps()}
    if(state.frame>=maxFrames()-1||(!master?.seeking&&master?.ended)){
      if(state.loops>=3){state.frame=maxFrames()-1;state.playing=false;$('#play').textContent='▶'}
      else{state.frame=0;state.loops++}
      syncRgb(true);render(true);
    }
    const refresh=now-state.lastMedia>=100;
    if(refresh){state.lastMedia=now;syncRgb();render(true)}
  }
  requestAnimationFrame(tick);
}
function render(refreshMedia=false){$('#timeline').value=state.frame;$('#clock').textContent=`帧 ${state.frame} / ${maxFrames()-1} · 循环 ${state.loops} / 3`;if(refreshMedia)renderMedia();drawCurves()}

// ============================== 深度与点云帧 ==============================
function renderMedia(){document.querySelectorAll('.method-cell[data-id]').forEach(cell=>{const video=cell.querySelector('.depth-view video'),histogram=cell.querySelector('.histogram-view canvas');if(video.readyState>=2)drawHistogram(video,histogram);video.onseeked=()=>{if(video.readyState>=2)drawHistogram(video,histogram)}})}
function renderPcds(){document.querySelectorAll('.method-cell[data-id]').forEach(cell=>{const id=cell.dataset.id,g=state.meta.groups[id];if(g.pcd&&!g.pcd_video)showPcd(cell.querySelector('.pcd-view'),id,state.pcdFrame)})}

// 当前深度图已按视角统一范围映射到 0–255，直接复用像素生成直方图。
function drawHistogram(image,canvas){const sample=document.createElement('canvas'),ctx=sample.getContext('2d',{willReadFrequently:true});sample.width=image.videoWidth||image.naturalWidth;sample.height=image.videoHeight||image.naturalHeight;ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,sample.width,sample.height).data,bins=new Uint32Array(64);for(let i=0;i<pixels.length;i+=4){const value=pixels[i];if(value>0)bins[Math.min(63,value>>2)]++}const out=canvas.getContext('2d'),max=Math.max(1,...bins);out.fillStyle='#090b0c';out.fillRect(0,0,canvas.width,canvas.height);for(let i=0;i<bins.length;i++){const height=bins[i]/max*(canvas.height-18),hue=240-i/63*240;out.fillStyle=`hsl(${hue} 78% 58%)`;out.fillRect(i*canvas.width/bins.length,canvas.height-height-12,Math.ceil(canvas.width/bins.length),height)}const gradient=out.createLinearGradient(0,0,canvas.width,0);for(let i=0;i<=6;i++)gradient.addColorStop(i/6,`hsl(${240-i*40} 78% 58%)`);out.fillStyle=gradient;out.fillRect(0,canvas.height-7,canvas.width,7)}
const pcdCache={},pcdScenes=new Map(),pcdTransforms=new Map();async function showPcd(viewport,id,frame){const key=`${id}:${frame}`;if(viewport.dataset.key===key)return;viewport.dataset.key=key;let canvas=viewport.querySelector('canvas');if(!canvas){viewport.innerHTML='<canvas width="360" height="155"></canvas>';canvas=viewport.querySelector('canvas');const ctx=canvas.getContext('2d');ctx.fillStyle='#090b0c';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#8f9a9b';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText('读取 PCD...',canvas.width/2,canvas.height/2)}try{const response=pcdCache[key]||await fetch(`/api/pcd?id=${id}&frame=${frame}`).then(r=>r.json());if(response.error)throw Error(response.error);pcdCache[key]=response;if(viewport.dataset.key!==key)return;viewport.title=`请求帧 ${frame} · ${response.file}`;createPcdScene(viewport,canvas,response.points,id.split('.')[0])}catch(error){if(viewport.dataset.key!==key)return;const ctx=canvas.getContext('2d');ctx.fillStyle='#090b0c';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#8f9a9b';ctx.font='11px system-ui';ctx.textAlign='center';ctx.fillText(error.message,canvas.width/2,canvas.height/2)}}

// 同一相机视角共享旋转和缩放。
function paintPcdView(view){
  pcdScenes.forEach(scene=>{if(scene.view===view)paintPcdScene(scene)});
}
function createPcdScene(viewport,canvas,points,view){
  if(!points.length)return;
  const center=[0,0,0];
  points.forEach(point=>{center[0]+=point[0];center[1]+=point[1];center[2]+=point[2]});
  center.forEach((_,i)=>center[i]/=points.length);
  const centered=points.map(point=>[point[0]-center[0],point[1]-center[1],point[2]-center[2]]);
  const radii=centered.map(point=>Math.hypot(...point)).sort((a,b)=>a-b),radius=radii[Math.floor((radii.length-1)*.90)]||1;
  const previous=pcdScenes.get(viewport);
  if(previous?.drag){canvas.releasePointerCapture(previous.drag.id);canvas.classList.remove('dragging')}
  if(!pcdTransforms.has(view))pcdTransforms.set(view,{x:0,y:0,zoom:1});
  const rotation=pcdTransforms.get(view);
  const scene={canvas,view,rotation,points:centered,scale:Math.min(canvas.width,canvas.height)*.72/radius,
    drag:null};
  canvas.title='拖拽：同视角点云同步旋转 · 滚轮：同视角同步缩放 · 双击：同视角复位';
  canvas.onpointerdown=event=>{
    if(event.button!==0||scene.drag)return;
    event.preventDefault();
    scene.drag={id:event.pointerId,x:event.clientX,y:event.clientY};
    canvas.setPointerCapture(event.pointerId);canvas.classList.add('dragging');
  };
  canvas.onpointermove=event=>{
    const drag=scene.drag;if(!drag||event.pointerId!==drag.id)return;
    rotation.y+=(event.clientX-drag.x)*.01;
    rotation.x-=(event.clientY-drag.y)*.01;
    drag.x=event.clientX;drag.y=event.clientY;
    paintPcdView(view);
  };
  const endDrag=event=>{
    if(scene.drag?.id!==event.pointerId)return;
    scene.drag=null;canvas.classList.remove('dragging');
    if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
  };
  canvas.onpointerup=endDrag;canvas.onpointercancel=endDrag;canvas.onlostpointercapture=endDrag;
  canvas.onwheel=event=>{event.preventDefault();rotation.zoom=clamp(rotation.zoom-event.deltaY*.001,.35,5);paintPcdView(view)};
  canvas.ondblclick=()=>{rotation.x=0;rotation.y=0;rotation.zoom=1;paintPcdView(view)};
  pcdScenes.set(viewport,scene);paintPcdScene(scene);
}
function paintPcdScene(scene){
  const {canvas,points}=scene,ctx=canvas.getContext('2d'),scale=scene.scale*scene.rotation.zoom;
  const cx=Math.cos(scene.rotation.x),sx=Math.sin(scene.rotation.x),cy=Math.cos(scene.rotation.y),sy=Math.sin(scene.rotation.y);
  ctx.fillStyle='#090b0c';ctx.fillRect(0,0,canvas.width,canvas.height);
  for(let i=0;i<points.length;i+=2){
    const[x,y,z]=points[i],yawX=cy*x+sy*z,yawZ=-sy*x+cy*z;
    const rotatedY=cx*y-sx*yawZ,rotatedZ=sx*y+cx*yawZ;
    const screenX=yawX+.28*rotatedZ,screenY=rotatedY-.12*rotatedZ;
    ctx.fillStyle=`hsl(${clamp(190-rotatedZ*35,25,210)} 65% 62%)`;
    ctx.fillRect(canvas.width/2+screenX*scale,canvas.height/2+screenY*scale,1.4,1.4);
  }
}

// ============================== 均值曲线 ==============================
function drawCurves(){views.forEach(view=>{const canvas=$(`#curve-${view}`);if(!canvas)return;const ctx=canvas.getContext('2d'),left=48,right=10,top=8,bottom=28,width=canvas.width-left-right,height=canvas.height-top-bottom,groups=methods.map(method=>state.meta.groups[`${view}.${method}`]).filter(g=>g?.available);if(!groups.length)return;const min=Math.min(...groups.map(g=>g.normalize_min)),max=Math.max(...groups.map(g=>g.normalize_max));ctx.clearRect(0,0,canvas.width,canvas.height);ctx.font='10px ui-monospace';ctx.fillStyle='#8f9a9b';ctx.strokeStyle='#303738';ctx.lineWidth=1;for(let i=0;i<=4;i++){const y=top+i*height/4,value=max-(max-min)*i/4;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(canvas.width-right,y);ctx.stroke();ctx.fillText(value.toFixed(0),2,y+3)}for(let i=0;i<=4;i++){const x=left+i*width/4,frame=Math.round((maxFrames()-1)*i/4);ctx.fillText(String(frame),x-8,canvas.height-8)}ctx.save();ctx.translate(11,top+height/2);ctx.rotate(-Math.PI/2);ctx.fillText('深度值',-16,0);ctx.restore();ctx.fillText('帧号',left+width/2-10,canvas.height-8);methods.forEach((method,index)=>{const g=state.meta.groups[`${view}.${method}`];if(!g?.means?.length)return;ctx.strokeStyle=['#e5b84b','#59b5bd','#d96b61'][index];ctx.lineWidth=2;ctx.beginPath();let started=false;g.means.forEach((mean,i)=>{if(!Number.isFinite(mean))return;const x=left+i/Math.max(1,g.means.length-1)*width,y=top+(max-mean)/(max-min||1)*height;started?ctx.lineTo(x,y):ctx.moveTo(x,y);started=true});ctx.stroke()});const x=left+state.frame/Math.max(1,maxFrames()-1)*width;ctx.strokeStyle='#ffffff99';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+height);ctx.stroke()})}
load();
