'use strict';

/* Recorded-pose WebGL renderer. It does not run dynamics, policy inference, or safety checks. */
window.SentinelNativeViewer = class {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', {antialias: true, alpha: false});
    if (!this.gl) throw new Error('当前浏览器无法创建 WebGL 视图。');
    this.yaw = -0.8;
    this.pitch = 0.45;
    this.distance = 1.5;
    this.target = [0, 0, 0.35];
    this.frameIndex = 0;
    this.meshes = [];
    this.record = null;
    this.trace = null;
    this._buildProgram();
    this._buildSceneGeometry();
    this._bindControls();
    this.gl.enable(this.gl.DEPTH_TEST);
    this.gl.clearColor(0.035, 0.055, 0.075, 1);
  }

  _buildProgram() {
    const gl = this.gl;
    const shader = (type, source) => {
      const value = gl.createShader(type);
      gl.shaderSource(value, source);
      gl.compileShader(value);
      if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
      return value;
    };
    this.program = gl.createProgram();
    gl.attachShader(this.program, shader(gl.VERTEX_SHADER, `
      attribute vec3 aPosition; attribute vec3 aNormal;
      uniform mat4 uModel; uniform mat4 uViewProjection;
      varying vec3 vNormal; varying vec3 vPosition;
      void main(){vec4 p=uModel*vec4(aPosition,1.0);vPosition=p.xyz;vNormal=mat3(uModel)*aNormal;gl_Position=uViewProjection*p;}
    `));
    gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform vec3 uColor; uniform vec3 uEye; uniform float uUnlit;
      varying vec3 vNormal; varying vec3 vPosition;
      void main(){
        if(uUnlit>0.5){gl_FragColor=vec4(uColor,1.0);return;}
        vec3 n=normalize(vNormal);if(!gl_FrontFacing)n=-n;
        vec3 l=normalize(vec3(-0.4,-0.6,1.0));float diffuse=max(dot(n,l),0.0);
        vec3 h=normalize(l+normalize(uEye-vPosition));float spec=pow(max(dot(n,h),0.0),30.0)*0.18;
        gl_FragColor=vec4(uColor*(0.42+0.54*diffuse)+vec3(spec),1.0);
      }
    `));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error('WebGL 着色程序不可用。');
    this.locations = {};
    for (const name of ['uModel', 'uViewProjection', 'uColor', 'uEye', 'uUnlit']) {
      this.locations[name] = gl.getUniformLocation(this.program, name);
    }
    this.position = gl.getAttribLocation(this.program, 'aPosition');
    this.normal = gl.getAttribLocation(this.program, 'aNormal');
    this.uintIndices = gl.getExtension('OES_element_index_uint');
  }

  _buildSceneGeometry() {
    this.floor = this._upload({
      positions: [-2,-2,-.006, 2,-2,-.006, 2,2,-.006, -2,-2,-.006, 2,2,-.006, -2,2,-.006],
      normals: Array(6).fill([0,0,1]).flat(), color: [.08,.115,.145],
    });
    const grid = [];
    for (let i = -20; i <= 20; i += 1) {
      const value = i * 0.1;
      grid.push(-2,value,-.004, 2,value,-.004, value,-2,-.004, value,2,-.004);
    }
    this.grid = this._upload({positions: grid, color: [.18,.25,.30]}, this.gl.LINES);
    this.marker = this._upload(this._sphereGeometry());
  }

  _upload(mesh, mode) {
    if (!Array.isArray(mesh.positions) || mesh.positions.length < 3 || mesh.positions.length > 3000000 || mesh.positions.length % 3 || !mesh.positions.every(Number.isFinite)) {
      throw new Error('模型网格缺少合法顶点。');
    }
    const gl = this.gl;
    const buffer = (data, target) => {
      const value = gl.createBuffer();
      gl.bindBuffer(target, value);
      gl.bufferData(target, data, gl.STATIC_DRAW);
      return value;
    };
    const output = {
      ...mesh,
      mode: mode === undefined ? gl.TRIANGLES : mode,
      vertex: buffer(new Float32Array(mesh.positions), gl.ARRAY_BUFFER),
      normal: buffer(new Float32Array(mesh.normals || Array(mesh.positions.length).fill(0)), gl.ARRAY_BUFFER),
      count: mesh.positions.length / 3,
    };
    if (Array.isArray(mesh.indices)) {
      if (mesh.indices.length > 4000000 || mesh.indices.some(index => !Number.isInteger(index) || index < 0 || index >= mesh.positions.length/3)) throw new Error('模型索引超出支持范围。');
      const large = mesh.positions.length / 3 > 65535;
      if (large && !this.uintIndices) throw new Error('浏览器不支持当前模型的大索引网格。');
      output.indexType = large ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;
      output.index = buffer(large ? new Uint32Array(mesh.indices) : new Uint16Array(mesh.indices), gl.ELEMENT_ARRAY_BUFFER);
      output.count = mesh.indices.length;
    }
    return output;
  }

  _delete(mesh) {
    if (!mesh) return;
    this.gl.deleteBuffer(mesh.vertex);
    this.gl.deleteBuffer(mesh.normal);
    if (mesh.index) this.gl.deleteBuffer(mesh.index);
  }

  _sphereGeometry() {
    const positions = [], normals = [], indices = [], rows = 12, columns = 18;
    for (let row = 0; row <= rows; row += 1) {
      for (let column = 0; column <= columns; column += 1) {
        const a = row * Math.PI / rows, b = column * 2 * Math.PI / columns;
        const normal = [Math.sin(a)*Math.cos(b), Math.sin(a)*Math.sin(b), Math.cos(a)];
        positions.push(...normal); normals.push(...normal);
      }
    }
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        const first = row * (columns + 1) + column, next = first + columns + 1;
        indices.push(first, next, first + 1, next, next + 1, first + 1);
      }
    }
    return {positions, normals, indices};
  }

  _positions(frame) {
    const value = frame?.bodyWorldPosition;
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== 'object') return [];
    return this.bodyNames.map(name => value[name]);
  }

  _rotations(frame) {
    const value = frame?.bodyWorldRotation;
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== 'object') return [];
    return this.bodyNames.map(name => value[name]);
  }

  setRecord(model, episode) {
    if (!model || !Array.isArray(model.meshes) || !model.meshes.length || model.meshes.length>512) throw new Error('任务模型不含受支持的可渲染网格。');
    if (!episode || !Array.isArray(episode.frames) || !episode.frames.length || episode.frames.length>5000) throw new Error('Episode 没有受支持的回放帧。');
    let bodyNames = model.bodyPositionOrder || model.body_position_order;
    if (!Array.isArray(bodyNames) && Array.isArray(model.bodies)) {
      const declaredIndices = model.bodies.map(body => Number(body.bodyIndex ?? body.body_index));
      if (!declaredIndices.length || declaredIndices.some(index => !Number.isInteger(index) || index < 0 || index > 4096)) {
        throw new Error('任务模型包含不合法的 bodyIndex。');
      }
      const highestDeclared = Math.max(...declaredIndices);
      bodyNames = Array.from({length: highestDeclared + 1}, (_, index) => `body-${index}`);
      for (const body of model.bodies) {
        const index = Number(body.bodyIndex ?? body.body_index);
        if (Number.isInteger(index) && index >= 0 && index < bodyNames.length && typeof body.name === 'string') bodyNames[index] = body.name;
      }
    }
    if (!Array.isArray(bodyNames) || !bodyNames.length || bodyNames.length>512 || bodyNames.some(name => typeof name !== 'string')) {
      throw new Error('任务模型缺少合法 body 索引。');
    }
    this.bodyNames = bodyNames;
    const poseFrames = episode.frames.map((frame,index) => ({frame,index})).filter(({frame}) => {
      return this._positions(frame).length === bodyNames.length && this._rotations(frame).length === bodyNames.length;
    });
    if (!poseFrames.length) throw new Error('Episode 不含与模型匹配的完整位姿。');
    const firstPositions = this._positions(poseFrames[0].frame), firstRotations = this._rotations(poseFrames[0].frame);
    const highestBody = Math.max(...model.meshes.map(mesh => Number(mesh.bodyIndex ?? mesh.body_index)));
    if (!Number.isInteger(highestBody) || highestBody < 0 || highestBody >= bodyNames.length ||
        firstPositions.length !== bodyNames.length || firstRotations.length !== bodyNames.length) {
      throw new Error('模型 body 映射与记录位姿不一致。');
    }
    const recordedModelIds = new Set(episode.frames.map(frame => frame.modelId).filter(Boolean));
    const declaredModelId = model.modelId || model.model_id;
    if (recordedModelIds.size > 1 || (declaredModelId && recordedModelIds.size && !recordedModelIds.has(declaredModelId))) {
      throw new Error('记录帧的 modelId 与任务模型不一致。');
    }
    for (const mesh of this.meshes) this._delete(mesh);
    this.meshes = model.meshes.map(mesh => this._upload({...mesh, bodyIndex: Number(mesh.bodyIndex ?? mesh.body_index)}));
    this._delete(this.trace);
    this.record = episode;
    this.traceBodyIndex = Number(model.endEffectorBodyIndex ?? model.end_effector_body_index ?? bodyNames.length - 1);
    if (!Number.isInteger(this.traceBodyIndex) || this.traceBodyIndex < 0 || this.traceBodyIndex >= bodyNames.length) {
      this.traceBodyIndex = bodyNames.length - 1;
    }
    this.record = episode;
    this.poseFrames = poseFrames;
    const tracePoints = poseFrames.map(({frame}) => this._positions(frame)[this.traceBodyIndex]).filter(this._point);
    this.trace = tracePoints.length > 1 ? this._upload({positions: tracePoints.flat(), color: [.25,.79,.76]}, this.gl.LINE_STRIP) : null;
    const mins = [Infinity,Infinity,Infinity], maxs = [-Infinity,-Infinity,-Infinity];
    for (const {frame} of poseFrames) for (const point of this._positions(frame)) {
      if (!this._point(point)) continue;
      for (let axis=0;axis<3;axis++) {
        mins[axis]=Math.min(mins[axis],point[axis]); maxs[axis]=Math.max(maxs[axis],point[axis]);
      }
    }
    this.target = [0,1,2].map(axis => (mins[axis] + maxs[axis]) / 2);
    let radius = .25;
    for (const {frame} of poseFrames) for (const point of this._positions(frame)) {
      if (this._point(point)) radius=Math.max(radius,Math.hypot(...point.map((value,axis)=>value-this.target[axis])));
    }
    this.distance = Math.max(.75, radius * 3.1);
    this.defaultView = {yaw: -0.8, pitch: 0.45, distance: this.distance, target: [...this.target]};
    this.frameIndex = 0;
    this.draw(0);
  }

  _point(value) {
    return Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
  }

  _rotation(value) {
    if (Array.isArray(value) && value.length === 9 && value.every(Number.isFinite)) return value;
    if (!Array.isArray(value) || value.length !== 4 || !value.every(Number.isFinite)) return [1,0,0,0,1,0,0,0,1];
    const [w,x,y,z] = value, n = Math.hypot(w,x,y,z) || 1, q0=w/n, q1=x/n, q2=y/n, q3=z/n;
    return [
      1-2*(q2*q2+q3*q3), 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2),
      2*(q1*q2+q0*q3), 1-2*(q1*q1+q3*q3), 2*(q2*q3-q0*q1),
      2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2),
    ];
  }

  _matrix(rotation, position, scale=1) {
    const r = rotation || [1,0,0,0,1,0,0,0,1], p = position || [0,0,0];
    return new Float32Array([r[0]*scale,r[3]*scale,r[6]*scale,0,r[1]*scale,r[4]*scale,r[7]*scale,0,r[2]*scale,r[5]*scale,r[8]*scale,0,...p,1]);
  }

  _multiply(a, b) {
    const output = new Float32Array(16);
    for (let column=0; column<4; column+=1) for (let row=0; row<4; row+=1) for (let k=0; k<4; k+=1) {
      output[column*4+row] += a[k*4+row] * b[column*4+k];
    }
    return output;
  }

  _camera(width, height) {
    const cosine=Math.cos(this.pitch), target=this.target;
    this.eye=[target[0]+this.distance*cosine*Math.sin(this.yaw),target[1]-this.distance*cosine*Math.cos(this.yaw),target[2]+this.distance*Math.sin(this.pitch)];
    const sub=(a,b)=>a.map((value,index)=>value-b[index]);
    const unit=value=>{const length=Math.hypot(...value)||1;return value.map(item=>item/length);};
    const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const dot=(a,b)=>a.reduce((sum,value,index)=>sum+value*b[index],0);
    const z=unit(sub(this.eye,target)), x=unit(cross([0,0,1],z)), y=cross(z,x);
    const view=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,this.eye),-dot(y,this.eye),-dot(z,this.eye),1]);
    const f=1/Math.tan(Math.PI/8), aspect=Math.max(.1,width/height), near=.01, far=Math.max(20,this.distance*12);
    const projection=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);
    return this._multiply(projection,view);
  }

  _drawMesh(mesh, matrix, color, unlit=false, count=mesh.count) {
    const gl=this.gl, locations=this.locations;
    gl.bindBuffer(gl.ARRAY_BUFFER,mesh.vertex);gl.enableVertexAttribArray(this.position);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);
    gl.bindBuffer(gl.ARRAY_BUFFER,mesh.normal);gl.enableVertexAttribArray(this.normal);gl.vertexAttribPointer(this.normal,3,gl.FLOAT,false,0,0);
    gl.uniformMatrix4fv(locations.uModel,false,matrix);gl.uniform3fv(locations.uColor,color||mesh.color||[.62,.68,.74]);gl.uniform1f(locations.uUnlit,unlit?1:0);
    if(mesh.index){gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.index);gl.drawElements(mesh.mode,count,mesh.indexType,0);}else gl.drawArrays(mesh.mode,0,count);
  }

  draw(index=this.frameIndex) {
    if (!this.record || this.gl.isContextLost()) return;
    this.frameIndex=Math.max(0,Math.min(this.record.frames.length-1,Number(index)||0));
    const gl=this.gl, bounds=this.canvas.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(bounds.width*dpr)), height=Math.max(1,Math.round(bounds.height*dpr));
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    gl.viewport(0,0,width,height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
    const viewProjection=this._camera(bounds.width||1,bounds.height||1);
    gl.uniformMatrix4fv(this.locations.uViewProjection,false,viewProjection);gl.uniform3fv(this.locations.uEye,this.eye);
    const identity=this._matrix();this._drawMesh(this.floor,identity,null,true);this._drawMesh(this.grid,identity,null,true);
    const poseEntry=this.poseFrames.findLast ? this.poseFrames.findLast(item=>item.index<=this.frameIndex) : [...this.poseFrames].reverse().find(item=>item.index<=this.frameIndex);
    const pose=poseEntry || this.poseFrames[0], frame=pose.frame, positions=this._positions(frame), rotations=this._rotations(frame);
    for(const mesh of this.meshes){
      const point=positions[mesh.bodyIndex];
      if(this._point(point))this._drawMesh(mesh,this._matrix(this._rotation(rotations[mesh.bodyIndex]),point));
    }
    if(this.trace){const visible=this.poseFrames.filter(item=>item.index<=this.frameIndex).length;this._drawMesh(this.trace,identity,[.25,.79,.76],true,Math.max(1,Math.min(this.trace.count,visible)));}
    const marker=positions[this.traceBodyIndex];
    if(this._point(marker))this._drawMesh(this.marker,this._matrix(null,marker,.012),[.20,.88,.73],true);
  }

  reset() {
    if (!this.defaultView) return;
    Object.assign(this,{...this.defaultView,target:[...this.defaultView.target]});
    this.draw();
  }

  clear() {
    this.record = null;
    this.poseFrames = [];
    const gl=this.gl, bounds=this.canvas.getBoundingClientRect(), dpr=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(1,Math.round(bounds.width*dpr)), height=Math.max(1,Math.round(bounds.height*dpr));
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    gl.viewport(0,0,width,height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  }

  _bindControls() {
    let pointer=null;
    this.canvas.addEventListener('pointerdown',event=>{pointer=[event.clientX,event.clientY];this.canvas.setPointerCapture(event.pointerId);});
    this.canvas.addEventListener('pointermove',event=>{
      if(!pointer)return;
      this.yaw-=(event.clientX-pointer[0])*.008;
      this.pitch=Math.max(-1.45,Math.min(1.45,this.pitch+(event.clientY-pointer[1])*.008));
      pointer=[event.clientX,event.clientY];this.draw();
    });
    const release=()=>{pointer=null;};
    this.canvas.addEventListener('pointerup',release);this.canvas.addEventListener('pointercancel',release);
    this.canvas.addEventListener('wheel',event=>{event.preventDefault();this.distance=Math.max(.12,Math.min(30,this.distance*Math.exp(event.deltaY*.001)));this.draw();},{passive:false});
    window.addEventListener('resize',()=>this.draw());
  }
};
