'use strict';

(function () {
  const TAU = Math.PI * 2;
  const BOX_FACES = [
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
    [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
    [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
  ];

  function finiteVector(value, length, fallback) {
    if (!Array.isArray(value) || value.length < length) return fallback.slice();
    const result = value.slice(0, length).map(Number);
    return result.every(Number.isFinite) ? result : fallback.slice();
  }

  function matrixPoint(matrix, point, position) {
    return [
      position[0] + matrix[0] * point[0] + matrix[1] * point[1] + matrix[2] * point[2],
      position[1] + matrix[3] * point[0] + matrix[4] * point[1] + matrix[5] * point[2],
      position[2] + matrix[6] * point[0] + matrix[7] * point[1] + matrix[8] * point[2],
    ];
  }

  function quaternionMatrix(value) {
    const q = finiteVector(value, 4, [1, 0, 0, 0]);
    const length = Math.hypot(...q) || 1;
    const [w, x, y, z] = q.map(component => component / length);
    return [
      1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ];
  }

  class SentinelViewer {
    constructor(canvas, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.mode = options.mode || 'mesh';
      this.yaw = -0.65;
      this.pitch = 0.5;
      this.zoom = 1;
      this.pan = [0, 0];
      this.vertices = [];
      this.triangles = [];
      this.frames = [];
      this.frame = 0;
      this.drag = null;

      this.resizeObserver = new ResizeObserver(() => this.draw());
      this.resizeObserver.observe(canvas);
      canvas.addEventListener('pointerdown', event => {
        this.drag = {x: event.clientX, y: event.clientY, button: event.button};
        canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener('pointermove', event => this.movePointer(event));
      canvas.addEventListener('pointerup', () => { this.drag = null; });
      canvas.addEventListener('contextmenu', event => event.preventDefault());
      canvas.addEventListener('wheel', event => {
        event.preventDefault();
        this.zoom = Math.max(0.25, Math.min(8, this.zoom * Math.exp(-event.deltaY * 0.001)));
        this.draw();
      }, {passive: false});
    }

    movePointer(event) {
      if (!this.drag) return;
      const dx = event.clientX - this.drag.x;
      const dy = event.clientY - this.drag.y;
      this.drag.x = event.clientX;
      this.drag.y = event.clientY;
      if (this.drag.button === 2) {
        this.pan[0] += dx;
        this.pan[1] += dy;
      } else {
        this.yaw += dx * 0.008;
        this.pitch = Math.max(-1.25, Math.min(1.25, this.pitch + dy * 0.008));
      }
      this.draw();
    }

    reset() {
      this.yaw = -0.65;
      this.pitch = 0.5;
      this.zoom = 1;
      this.pan = [0, 0];
      this.draw();
    }

    setGeometry(geometry = {}) {
      this.vertices = Array.isArray(geometry.vertices) ? geometry.vertices.slice() : [];
      this.triangles = Array.isArray(geometry.triangles) ? geometry.triangles.slice() : [];
      if (!this.vertices.length && Array.isArray(geometry.primitives)) {
        geometry.primitives.forEach(primitive => this.addPrimitive(primitive));
      }
      this.mode = 'mesh';
      this.frame = 0;
      this.fit();
    }

    addPrimitive(primitive) {
      const position = finiteVector(primitive.position, 3, [0, 0, 0]);
      const rotation = finiteVector(primitive.rotation, 9, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
      const dimensions = Array.isArray(primitive.dimensions) ? primitive.dimensions.map(Number) : [];
      const mesh = {vertices: [], triangles: []};
      if (primitive.type === 'box' && dimensions.length >= 3) {
        this.boxMesh(mesh, dimensions);
      } else if (primitive.type === 'sphere' && dimensions.length >= 1) {
        this.sphereMesh(mesh, dimensions[0]);
      } else if (primitive.type === 'cylinder' && dimensions.length >= 2) {
        this.cylinderMesh(mesh, dimensions[0], dimensions[1]);
      } else if (primitive.type === 'capsule' && dimensions.length >= 2) {
        this.capsuleMesh(mesh, dimensions[0], dimensions[1]);
      } else {
        return;
      }
      const base = this.vertices.length;
      this.vertices.push(...mesh.vertices.map(point => matrixPoint(rotation, point, position)));
      this.triangles.push(...mesh.triangles.map(face => face.map(index => index + base)));
    }

    boxMesh(mesh, dimensions) {
      const half = dimensions.slice(0, 3).map(value => value / 2);
      for (const z of [-1, 1]) {
        for (const y of [-1, 1]) {
          for (const x of [-1, 1]) mesh.vertices.push([x * half[0], y * half[1], z * half[2]]);
        }
      }
      mesh.triangles.push(...BOX_FACES.map(face => face.slice()));
    }

    sphereMesh(mesh, radius, latitudeSteps = 12, longitudeSteps = 18, offsetZ = 0) {
      const start = mesh.vertices.length;
      for (let latitude = 0; latitude <= latitudeSteps; latitude += 1) {
        const theta = Math.PI * latitude / latitudeSteps;
        for (let longitude = 0; longitude < longitudeSteps; longitude += 1) {
          const phi = TAU * longitude / longitudeSteps;
          mesh.vertices.push([
            radius * Math.sin(theta) * Math.cos(phi),
            radius * Math.sin(theta) * Math.sin(phi),
            offsetZ + radius * Math.cos(theta),
          ]);
        }
      }
      for (let latitude = 0; latitude < latitudeSteps; latitude += 1) {
        for (let longitude = 0; longitude < longitudeSteps; longitude += 1) {
          const next = (longitude + 1) % longitudeSteps;
          const a = start + latitude * longitudeSteps + longitude;
          const b = start + latitude * longitudeSteps + next;
          const c = start + (latitude + 1) * longitudeSteps + longitude;
          const d = start + (latitude + 1) * longitudeSteps + next;
          mesh.triangles.push([a, c, b], [b, c, d]);
        }
      }
    }

    cylinderMesh(mesh, radius, length, steps = 18) {
      const start = mesh.vertices.length;
      for (const z of [-length / 2, length / 2]) {
        for (let index = 0; index < steps; index += 1) {
          const angle = TAU * index / steps;
          mesh.vertices.push([radius * Math.cos(angle), radius * Math.sin(angle), z]);
        }
      }
      const bottomCenter = mesh.vertices.push([0, 0, -length / 2]) - 1;
      const topCenter = mesh.vertices.push([0, 0, length / 2]) - 1;
      for (let index = 0; index < steps; index += 1) {
        const next = (index + 1) % steps;
        mesh.triangles.push(
          [start + index, start + steps + index, start + next],
          [start + next, start + steps + index, start + steps + next],
          [bottomCenter, start + next, start + index],
          [topCenter, start + steps + index, start + steps + next],
        );
      }
    }

    capsuleMesh(mesh, radius, cylinderLength, latitudeSteps = 8, longitudeSteps = 18) {
      const rings = [];
      for (let index = 0; index <= latitudeSteps; index += 1) {
        const angle = -Math.PI / 2 + (Math.PI / 2) * index / latitudeSteps;
        rings.push({radial: radius * Math.cos(angle), z: -cylinderLength / 2 + radius * Math.sin(angle)});
      }
      for (let index = 0; index <= latitudeSteps; index += 1) {
        const angle = (Math.PI / 2) * index / latitudeSteps;
        rings.push({radial: radius * Math.cos(angle), z: cylinderLength / 2 + radius * Math.sin(angle)});
      }
      const start = mesh.vertices.length;
      for (const ring of rings) {
        for (let longitude = 0; longitude < longitudeSteps; longitude += 1) {
          const angle = TAU * longitude / longitudeSteps;
          mesh.vertices.push([
            ring.radial * Math.cos(angle),
            ring.radial * Math.sin(angle),
            ring.z,
          ]);
        }
      }
      for (let ring = 0; ring < rings.length - 1; ring += 1) {
        for (let longitude = 0; longitude < longitudeSteps; longitude += 1) {
          const next = (longitude + 1) % longitudeSteps;
          const a = start + ring * longitudeSteps + longitude;
          const b = start + ring * longitudeSteps + next;
          const c = start + (ring + 1) * longitudeSteps + longitude;
          const d = start + (ring + 1) * longitudeSteps + next;
          mesh.triangles.push([a, c, b], [b, c, d]);
        }
      }
    }

    setTrace(trace = []) {
      this.frames = Array.isArray(trace) ? trace.filter(Boolean) : [];
      this.mode = 'trace';
      this.frame = Math.max(0, this.frames.length - 1);
      this.fit();
    }

    setFrame(index) {
      this.frame = Math.max(0, Math.min(this.frames.length - 1, Number(index) || 0));
      this.draw();
    }

    fit() {
      this.zoom = 1;
      this.pan = [0, 0];
      this.draw();
    }

    size() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width;
        this.canvas.height = height;
      }
      return {width, height};
    }

    rotate(point) {
      const [x, y, z] = point.map(Number);
      const cy = Math.cos(this.yaw);
      const sy = Math.sin(this.yaw);
      const cp = Math.cos(this.pitch);
      const sp = Math.sin(this.pitch);
      const horizontal = cy * x - sy * y;
      const depth = sy * x + cy * y;
      return [horizontal, cp * z - sp * depth, sp * z + cp * depth];
    }

    bounds(points) {
      if (!points.length) return {center: [0, 0, 0], scale: 1};
      const low = [Infinity, Infinity, Infinity];
      const high = [-Infinity, -Infinity, -Infinity];
      for (const point of points) {
        for (let axis = 0; axis < 3; axis += 1) {
          const value = Number(point[axis]) || 0;
          low[axis] = Math.min(low[axis], value);
          high[axis] = Math.max(high[axis], value);
        }
      }
      const center = low.map((value, axis) => (value + high[axis]) / 2);
      const span = Math.max(0.001, ...low.map((value, axis) => high[axis] - value));
      return {center, scale: 1 / span};
    }

    project(point, bounds, width, height) {
      const local = point.map((value, axis) => (Number(value) || 0) - bounds.center[axis]);
      const rotated = this.rotate(local);
      const scale = Math.min(width, height) * 0.62 * bounds.scale * this.zoom;
      return [
        width / 2 + this.pan[0] + rotated[0] * scale,
        height / 2 + this.pan[1] - rotated[1] * scale,
        rotated[2],
      ];
    }

    drawGrid(ctx, width, height) {
      ctx.save();
      ctx.strokeStyle = '#dce7e4';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.58;
      const horizon = height * 0.64;
      const step = Math.max(28, Math.min(width, height) / 9);
      for (let index = -12; index <= 12; index += 1) {
        ctx.beginPath();
        ctx.moveTo(width / 2 + index * step, horizon);
        ctx.lineTo(width / 2 + index * step * 2.8, height);
        ctx.stroke();
      }
      for (let index = 0; index < 8; index += 1) {
        const y = horizon + (height - horizon) * (index / 8) ** 1.65;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      ctx.restore();
    }

    draw() {
      const {width, height} = this.size();
      const ctx = this.ctx;
      ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, '#f7fbfb');
      gradient.addColorStop(1, '#eef4f2');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, width, height);
      this.drawGrid(ctx, width, height);
      if (this.mode === 'mesh') this.drawMesh(ctx, width, height);
      else this.drawTrace(ctx, width, height);
    }

    drawTriangles(ctx, vertices, triangles, bounds, width, height, color) {
      const projected = vertices.map(vertex => this.project(vertex, bounds, width, height));
      const faces = triangles
        .filter(face => Array.isArray(face) && face.length >= 3 && face.every(index => projected[index]))
        .map(face => ({points: face.map(index => projected[index]), depth: face.reduce((sum, index) => sum + projected[index][2], 0) / 3}))
        .sort((a, b) => b.depth - a.depth);
      ctx.lineJoin = 'round';
      for (const face of faces) {
        const alpha = 0.5 + Math.max(-0.18, Math.min(0.18, face.depth * bounds.scale));
        ctx.beginPath();
        ctx.moveTo(face.points[0][0], face.points[0][1]);
        face.points.slice(1).forEach(point => ctx.lineTo(point[0], point[1]));
        ctx.closePath();
        ctx.fillStyle = `rgba(${color},${alpha})`;
        ctx.fill();
        ctx.strokeStyle = 'rgba(10,75,82,.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    drawMesh(ctx, width, height) {
      if (!this.vertices.length) return;
      const bounds = this.bounds(this.vertices);
      this.drawTriangles(ctx, this.vertices, this.triangles, bounds, width, height, '39,151,146');
    }

    tracePoint(frame, key) {
      const value = frame?.[key] || frame?.feedback?.[key];
      return Array.isArray(value) && value.length >= 3 ? finiteVector(value, 3, [0, 0, 0]) : null;
    }

    traceQuaternion(frame) {
      const qpos = frame?.qpos || frame?.feedback?.qpos;
      return Array.isArray(qpos) && qpos.length >= 10 ? finiteVector(qpos.slice(-4), 4, [1, 0, 0, 0]) : [1, 0, 0, 0];
    }

    bodyMesh(position, dimensions, rotation) {
      const mesh = {vertices: [], triangles: []};
      this.boxMesh(mesh, dimensions);
      mesh.vertices = mesh.vertices.map(point => matrixPoint(rotation, point, position));
      return mesh;
    }

    drawTrace(ctx, width, height) {
      if (!this.frames.length) return;
      const points = [];
      for (const frame of this.frames) {
        const tray = this.tracePoint(frame, 'position') || this.tracePoint(frame, 'qpos');
        const payload = this.tracePoint(frame, 'payload_position');
        if (tray) points.push(tray);
        if (payload) points.push(payload);
      }
      const bounds = this.bounds(points);
      for (const [key, color] of [['position', '#129caa'], ['payload_position', '#ee9a3d']]) {
        const path = [];
        for (let index = 0; index <= this.frame; index += 1) {
          const point = this.tracePoint(this.frames[index], key)
            || (key === 'position' ? this.tracePoint(this.frames[index], 'qpos') : null);
          if (point) path.push(this.project(point, bounds, width, height));
        }
        if (!path.length) continue;
        ctx.beginPath();
        path.forEach((point, index) => index ? ctx.lineTo(point[0], point[1]) : ctx.moveTo(point[0], point[1]));
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.75;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      const current = this.frames[this.frame];
      const trayPosition = this.tracePoint(current, 'position') || this.tracePoint(current, 'qpos');
      const payloadPosition = this.tracePoint(current, 'payload_position');
      if (trayPosition) {
        const tray = this.bodyMesh(trayPosition, [0.24, 0.20, 0.016], [1, 0, 0, 0, 1, 0, 0, 0, 1]);
        this.drawTriangles(ctx, tray.vertices, tray.triangles, bounds, width, height, '18,156,170');
      }
      if (payloadPosition) {
        const payload = this.bodyMesh(payloadPosition, [0.04, 0.036, 0.044], quaternionMatrix(this.traceQuaternion(current)));
        this.drawTriangles(ctx, payload.vertices, payload.triangles, bounds, width, height, '238,154,61');
      }
    }
  }

  window.SentinelViewer = SentinelViewer;
})();
