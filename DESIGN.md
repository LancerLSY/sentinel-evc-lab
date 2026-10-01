# Design

## Source of truth
- Status: Active
- Last refreshed: 2026-10-01
- Primary product surfaces: installable CLI, macOS desktop App, local browser workbench.
- Evidence reviewed: existing web/index.html/app.js/styles.css, product_contracts.md, physics_ssh_design.md, physics_review_zh.md, user request for attractive UI, visualizations, arm and 3D model entrypoints.

## Brand
- Personality: precise, calm engineering instrument; polished, spacious, useful.
- Trust signals: actual execution state, measured results, source/profile labels, immutable asset digests, signed evidence.
- Avoid: fabricated metrics, animated fake telemetry, generic neon AI styling, presentation-only controls.

## Product goals
- Goals: install once; open CLI or App; run numeric and real MuJoCo experiments; import 3D assets and inspect them; configure/diagnose robot adapters; inspect evidence.
- Non-goals: unverified arbitrary robot motion, claim imported meshes are a collision certificate, pretend OBJ preview is physics.
- Success signals: user can reach all four entrypoints, complete actual experiment and replay results, import a model and see its geometry, see honest disconnected/unsupported states.

## Personas and jobs
- Primary personas: VLA/robotics researchers and engineers.
- User jobs: experiment locally/over SSH, inspect contact/trajectory evidence, integrate robot model/controller.
- Key contexts of use: macOS desktop 1280–1600px; laptop 1024px; fallback browser, occasional mobile inspection.

## Information architecture
- Primary navigation: 实验工作台 / 三维实验 / 模型资产 / 机械臂接入 / 实验计划.
- Core routes/screens: current numeric runs retain stop/resume/export; real physics job list/start/result; model upload/list/preview/check; robot profile and read-only diagnostic.
- Content hierarchy: page title and scope, primary action, meaningful visual, current status/actual metrics, detailed records.

## Design principles
- Shared engine: CLI and App call the same modules, no duplicate execution logic.
- Evidence before decoration: empty/loading/failure states are first class; no guessed numbers.
- Direct workflows: import -> inspect -> physics validation, create -> run -> verify -> download.
- Tradeoffs: first desktop release uses native WebKit shell with local Python service; visual model import is separate from motion authorization.

## Visual language
- Color: graphite/navy sidebar, warm off-white canvas, white panels, restrained teal/blue primary, amber for incomplete scientific validation, red for actual errors.
- Typography: native system Chinese and sans-serif; clear hierarchy 28–32px titles/14px body/12px captions; monospace only IDs/technical details.
- Spacing/layout rhythm: 8px base, 24–32px panel/page padding, 16–24px gaps; wide three-dimensional stage.
- Shape/radius/elevation: 12–16px panels, 8px controls, thin neutral borders and quiet shadows.
- Motion: short 120–180ms state transitions; interaction-driven 3D orbit; pause animation offscreen.
- Imagery/iconography: local inline SVG icons, real mesh geometry/recorded trajectories. No CDN or stock images.

## Components
- Existing components to reuse: forms, candidate table, actual trace charts, signed export, events/replay.
- New/changed components: sidebar sections, experiment status chips, physics job cards, orbitable mesh viewport, asset importer, robot diagnostic panel, concise result metrics.
- Variants and states: disconnected, configured, read-only connected, unsupported motion, running, finished with failed scientific gate.
- Token/component ownership: web/styles.css tokens; web/app.js UI; viewer.js owns reusable dependency-free Canvas visualization.

## Accessibility
- Target standard: WCAG 2.2 AA design target; validate key interaction/focus/contrast rather than claim certification.
- Keyboard/focus behavior: visible focus, labelled forms, native buttons/ranges, keyboard-accessible tabs and viewer reset.
- Contrast/readability: dark text on light panels; state communicated with words/icons as well as color.
- Screen-reader semantics: nav/main/sections, meaningful button labels, live status region, canvas alternative summary.
- Reduced motion and sensory considerations: respect prefers-reduced-motion; avoid flashing or compulsory automatic camera spin.

## Responsive behavior
- Supported breakpoints/devices: desktop >=1200, compact 850–1199, narrow <850.
- Layout adaptations: compact/sidebar -> top nav; form columns stack; viewport retains usable height; tables scroll horizontally.
- Touch/hover differences: explicit controls, 40px+ targets, no hover-only functionality.

## Interaction states
- Loading: actual job status and pending request; disable duplicate submission.
- Empty: explain next action without mock rows or invented asset/robot counts.
- Error: retain input, show server rejection, permit retry.
- Success: actual imported hash/geometry, completed evidence and acceptance results.
- Disabled: explain unavailable hardware motion or format validation.
- Offline/slow network: visible connection issue; preserve current inspected result, bounded requests.

## Content voice
- Tone: concise Chinese action labels and plain descriptions.
- Terminology: 模型预览 / MuJoCo物理检查 / 只读诊断 / 已完成但验收未全通过.
- Microcopy rules: distinguish mesh preview, mechanical model simulation, trusted fixture execution, and physical hardware. No speculative safety promise.

## Implementation constraints
- Framework/styling system: standard Python library HTTP and vanilla HTML/CSS/JS; native macOS AppKit/WKWebView shell; no new frontend/runtime framework.
- Design-token constraints: extend repo-native CSS variables; keep all assets offline.
- Performance constraints: bounded mesh/model input, finite geometry; background jobs rather than block UI or Executor; bounded traces/render work.
- Compatibility constraints: Python 3.10+, optional MuJoCo 3.14; desktop package must explain Python prerequisite/distribution status honestly.
- Test/screenshot expectations: actual browser workflows, desktop launch smoke, responsive inspection, screenshots with actual assets/results, preserve historical tests.

## Open questions
- [ ] Target physical robot brand/model and driver protocol: user pending; motion remains disabled until reviewed adapter/device test.
- [ ] Preferred external model format: user pending; implement OBJ/STL/MJCF/URDF entrypoints with honest per-format capabilities.
- [ ] Distribution signing/notarization and other operating systems: later release; do not claim notarized standalone cross-platform binary.
