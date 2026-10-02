/* Native WebGL renderer for source-bound UR5e visual meshes. No dynamics run here. */
"use strict";
window.SentinelReplayViewer = class {
  constructor(canvas, model) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl", { antialias:true, alpha:false });
    if (!this.gl) throw new Error("WebGL unavailable");
    this.model = model;
    this.yaw = -.85; this.pitch = .46; this.distance = 1.85;
    this.target = [-.19,.25,.34];
    const gl = this.gl;
    const shader = (type, source) => {
      const s=gl.createShader(type); gl.shaderSource(s,source); gl.compileShader(s);
      if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    this.program=gl.createProgram();
    gl.attachShader(this.program,shader(gl.VERTEX_SHADER,`attribute vec3 aPosition; attribute vec3 aNormal;
      uniform mat4 uModel; uniform mat4 uViewProjection;
      varying vec3 vNormal; varying vec3 vPosition;
      void main(){vec4 p=uModel*vec4(aPosition,1.0);vPosition=p.xyz;vNormal=mat3(uModel)*aNormal;gl_Position=uViewProjection*p;}`));
    gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,`precision mediump float;
      uniform vec3 uColor;uniform vec3 uEye;uniform float uUnlit;
      varying vec3 vNormal;varying vec3 vPosition;
      void main(){if(uUnlit>0.5){gl_FragColor=vec4(uColor,1.0);return;}vec3 n=normalize(vNormal);if(!gl_FrontFacing)n=-n;
      vec3 l=normalize(vec3(-0.4,-0.6,1.0));float diffuse=max(dot(n,l),0.0);
      vec3 h=normalize(l+normalize(uEye-vPosition));float spec=pow(max(dot(n,h),0.0),38.0)*0.22;
      vec3 shaded=uColor*(0.48+0.48*diffuse)+vec3(spec);
      gl_FragColor=vec4(mix(shaded,uColor,uUnlit),1.0);}`));
    gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error("WebGL program unavailable");
    this.locations={};
    for(const name of ["uModel","uViewProjection","uColor","uEye","uUnlit"])this.locations[name]=gl.getUniformLocation(this.program,name);
    this.position=gl.getAttribLocation(this.program,"aPosition");this.normal=gl.getAttribLocation(this.program,"aNormal");
    this.uintIndices=gl.getExtension("OES_element_index_uint");
    this.meshes=model.meshes.map(m=>this.upload(m));
    this.sphere=this.upload(this.sphereGeometry());
    this.floor=this.upload({positions:[-1.05,-.55,-.008,1.05,-.55,-.008,1.05,1.2,-.008,-1.05,-.55,-.008,1.05,1.2,-.008,-1.05,1.2,-.008],normals:Array(6).fill([0,0,1]).flat(),color:[.92,.94,.97]});
    const grid=[];for(let i=-10;i<=12;i++){let t=i*.1;grid.push(-1,t,-.005,1,t,-.005,t,-.5,-.005,t,1.2,-.005);}
    this.grid=this.upload({positions:grid,color:[.80,.84,.89]},gl.LINES);
    this.trace=null;this.completedTrace=null;
    gl.enable(gl.DEPTH_TEST);gl.clearColor(.957,.969,.988,1);
  }
  upload(mesh,mode) {
    const gl=this.gl;
    const buffer=(data,target)=>{const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b;};
    const out={...mesh,mode:mode===undefined?gl.TRIANGLES:mode,vertex:buffer(new Float32Array(mesh.positions),gl.ARRAY_BUFFER),normal:buffer(new Float32Array(mesh.normals||Array(mesh.positions.length).fill(0)),gl.ARRAY_BUFFER),count:mesh.positions.length/3};
    if(mesh.indices){const large=mesh.positions.length/3>65535;
      if(large&&!this.uintIndices)throw new Error("Large mesh indices unsupported");
      out.indexType=large?gl.UNSIGNED_INT:gl.UNSIGNED_SHORT;
      out.index=buffer(large?new Uint32Array(mesh.indices):new Uint16Array(mesh.indices),gl.ELEMENT_ARRAY_BUFFER);out.count=mesh.indices.length;
    }return out;
  }
  sphereGeometry() {
    const positions=[],normals=[],indices=[],rows=20,cols=32;
    for(let i=0;i<=rows;i++)for(let j=0;j<=cols;j++){const a=i*Math.PI/rows,b=j*2*Math.PI/cols;const n=[Math.sin(a)*Math.cos(b),Math.sin(a)*Math.sin(b),Math.cos(a)];positions.push(...n);normals.push(...n);}
    for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){let a=i*(cols+1)+j,b=a+cols+1;indices.push(a,b,a+1,b,b+1,a+1);}
    return {positions,normals,indices};
  }
  matrix(rotation,position,scale=1) {
    const r=rotation||[1,0,0,0,1,0,0,0,1],p=position||[0,0,0];
    return new Float32Array([r[0]*scale,r[3]*scale,r[6]*scale,0,r[1]*scale,r[4]*scale,r[7]*scale,0,r[2]*scale,r[5]*scale,r[8]*scale,0,...p,1]);
  }
  multiply(a,b){const out=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)out[c*4+r]+=a[k*4+r]*b[c*4+k];return out;}
  camera(width,height) {
    const c=Math.cos(this.pitch),t=this.target;
    this.eye=[t[0]+this.distance*c*Math.sin(this.yaw),t[1]-this.distance*c*Math.cos(this.yaw),t[2]+this.distance*Math.sin(this.pitch)];
    const sub=(a,b)=>a.map((v,i)=>v-b[i]),unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/n);},cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
    const z=unit(sub(this.eye,t)),x=unit(cross([0,0,1],z)),y=cross(z,x);
    const view=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,this.eye),-dot(y,this.eye),-dot(z,this.eye),1]);
    const f=1/Math.tan(Math.PI/9),aspect=width/height,near=.025,far=15;
    const projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
    this.viewProjection=this.multiply(projection,view);
  }
  setCase(record) {
    const gl=this.gl;for(const m of [this.trace,this.completedTrace])if(m){gl.deleteBuffer(m.vertex);gl.deleteBuffer(m.normal);}
    this.record=record;
    this.tracePositions=record.frames.flatMap(f=>f.bodyWorldPositionM[6]);
    const path=[];
    for(let i=1;i<record.frames.length;i++){
      const a=record.frames[i-1].bodyWorldPositionM[6],b=record.frames[i].bodyWorldPositionM[6],d=b.map((v,j)=>v-a[j]),len=Math.hypot(...d);
      const axis=len>1e-10?d.map(v=>v/len):[0,0,1],up=Math.abs(axis[2])<.9?[0,0,1]:[1,0,0];
      const cross=(u,v)=>[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]],u=cross(axis,up),un=Math.hypot(...u),x=u.map(v=>v/un),y=cross(axis,x);
      const point=(p,k)=>p.map((v,j)=>v+.0025*(x[j]*Math.cos(k*Math.PI/6)+y[j]*Math.sin(k*Math.PI/6)));
      for(let k=0;k<12;k++)for(const p of [point(a,k),point(b,k),point(a,k+1),point(b,k),point(b,k+1),point(a,k+1)])path.push(...p);
    }
    this.trace=this.upload({positions:path,color:[.62,.70,.81]});
    this.completedTrace=this.upload({positions:path,color:[.10,.40,.83]});
    this.poseFrames=this.model.cases.find(c=>c.rootId===record.rootId).frames;
    this.fitPoints=record.frames.flatMap(f=>f.bodyWorldPositionM).flatMap(p=>[[-.08,-.08,-.08],[.08,.08,.08]].map(d=>p.map((v,i)=>v+d[i])));
    for(const o of record.obstacles)if(Math.hypot(...o.centerWorldM)<1.5)for(const sign of [-1,1])this.fitPoints.push(o.centerWorldM.map(v=>v+sign*o.radiusM));
    this.target=[0,1,2].map(i=>(Math.min(...this.fitPoints.map(p=>p[i]))+Math.max(...this.fitPoints.map(p=>p[i])))/2);
    this.autoFit=true;
  }
  drawMesh(mesh,matrix,color,unlit=false,count=mesh.count) {
    const gl=this.gl,l=this.locations;
    gl.bindBuffer(gl.ARRAY_BUFFER,mesh.vertex);gl.enableVertexAttribArray(this.position);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,mesh.normal);gl.enableVertexAttribArray(this.normal);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,0,0);
    gl.uniformMatrix4fv(l.uModel,false,matrix);gl.uniform3fv(l.uColor,color||mesh.color||[.7,.75,.82]);gl.uniform1f(l.uUnlit,unlit?1:0);
    if(mesh.index){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.index);gl.drawElements(mesh.mode,count,mesh.indexType,0);}else gl.drawArrays(mesh.mode,0,count);
  }
  draw(frameIndex,showTrace=true) {
    if(!this.record||this.gl.isContextLost())return;
    const gl=this.gl,box=this.canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);
    const width=Math.round(box.width*dpr),height=Math.round(box.height*dpr);if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
    if(this.autoFit&&this.fitPoints){
      const c=Math.cos(this.pitch),z=[c*Math.sin(this.yaw),-c*Math.cos(this.yaw),Math.sin(this.pitch)],x=[Math.cos(this.yaw),Math.sin(this.yaw),0],y=[-Math.sin(this.pitch)*Math.sin(this.yaw),Math.sin(this.pitch)*Math.cos(this.yaw),c],tan=Math.tan(Math.PI/9),aspect=box.width/box.height;
      this.distance=Math.max(1.35,...this.fitPoints.map(p=>{const q=p.map((v,i)=>v-this.target[i]),dot=a=>q.reduce((s,v,i)=>s+v*a[i],0);return Math.max(Math.abs(dot(x))/(tan*aspect),Math.abs(dot(y))/tan)*1.12+dot(z);}));
    }
    this.camera(box.width,box.height);gl.uniformMatrix4fv(this.locations.uViewProjection,false,this.viewProjection);gl.uniform3fv(this.locations.uEye,this.eye);
    const identity=this.matrix();this.drawMesh(this.floor,identity,null,true);this.drawMesh(this.grid,identity,null,true);
    const frame=this.record.frames[frameIndex],pose=this.poseFrames[frameIndex];
    for(const mesh of this.meshes)this.drawMesh(mesh,this.matrix(pose.bodyWorldRotation[mesh.bodyIndex],frame.bodyWorldPositionM[mesh.bodyIndex]));
    for(const o of this.record.obstacles)this.drawMesh(this.sphere,this.matrix(null,o.centerWorldM,o.radiusM),[.92,.39,.13]);
    if(showTrace){this.drawMesh(this.trace,identity,null,true);this.drawMesh(this.completedTrace,identity,null,true,frameIndex*72);
      for(const [i,color] of [[0,[.11,.42,.84]],[this.record.frames.length-1,[.42,.49,.60]]])this.drawMesh(this.sphere,this.matrix(null,this.record.frames[i].bodyWorldPositionM[6],.008),color,true);
    }
    this.drawMesh(this.sphere,this.matrix(null,frame.bodyWorldPositionM[6],.011),[.0,.55,.75],true);
  }
  project(point){if(!this.viewProjection)return null;const m=this.viewProjection,p=[...point,1],q=[];for(let r=0;r<4;r++)q[r]=p.reduce((s,v,c)=>s+m[c*4+r]*v,0);if(q[3]<=0)return null;return {x:(q[0]/q[3]+1)*.5,y:(1-q[1]/q[3])*.5,visible:Math.abs(q[0]/q[3])<.94&&Math.abs(q[1]/q[3])<.90&&q[2]/q[3]<1};}
  preset(name){this.autoFit=true;if(name==="top"){this.yaw=0;this.pitch=1.52;this.distance=1.9;}else if(name==="side"){this.yaw=-Math.PI/2;this.pitch=.10;this.distance=1.85;}else{this.yaw=-.85;this.pitch=.46;this.distance=1.85;}}
};
