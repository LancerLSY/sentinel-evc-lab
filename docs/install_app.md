# 安装入口与 macOS App

Sentinel EVC 提供同一套 Python 引擎的两个入口：命令行用于安装、自动化和远端实验，macOS App 用原生 AppKit/WebKit 窗口承载本地工作台。App 不复制执行逻辑，也不把浏览器页面伪装成独立计算引擎。

## 一次安装

在交互终端中不传目录即可打开安装向导：

```bash
./tools/install.sh
# 已有可用 Python 环境时：
python -m sentinel_evc install --interactive
```

向导提供完整版/轻量版选择、安装目录、macOS App 开关和安装配置预览。
输入数字选择，回车接受默认值，`q` 或 `Ctrl+C` 退出。已有目录会提示重新选择，
不会覆盖。安装时按实际阶段显示状态和耗时；依赖下载与编译输出写入本地日志，
失败显示所在阶段及日志位置。完整日志保留，未发布的临时安装自动清理。

完成后可以打开 App 或浏览器工作台、查看 SSH 实验命令，或直接退出。
SSH 菜单只展示接入命令，不读取凭据或修改 SSH 配置。

Windows 终端向导：

```powershell
.\tools\install.ps1
# 可预填默认目录与类型，仍由向导选择：
.\tools\install.ps1 -Interactive -Target "$env:LOCALAPPDATA\Sentinel-EVC" -Profile core
```

无人值守安装保留显式目录方式；不传目录的向导要求交互终端，
重定向输入时会直接提示退出，不会卡住自动化作业。

前提是 Python 3.10+。默认 `all` profile 安装核心依赖和 MuJoCo 物理依赖，并在 macOS 构建 App：

```bash
./tools/install.sh "$HOME/Applications/Sentinel-EVC"
```

只安装轻量核心：

```bash
SENTINEL_PROFILE=core ./tools/install.sh "$HOME/Applications/Sentinel-EVC-Core"
```

Windows 当前提供 CLI 安装入口，不声称提供原生 App：

```powershell
.\tools\install.ps1 -Target "$env:LOCALAPPDATA\Sentinel-EVC" -Profile all
```

安装目标必须尚不存在。安装器先在同级临时目录完成虚拟环境、依赖、公开源码副本、启动器和 App 构建，全部成功后才发布目标；失败不会留下看似可用的半安装目录。公开源码副本包含 `src/`、`tests/`、`pyproject.toml`、`README.md` 和 `LICENSE`，供 SSH 作业与证据清单绑定使用，不包含 Git 历史、运行结果或私有配置。

安装后的命令位于 `bin/sentinel-evc`（Windows 为 `bin\sentinel-evc.cmd`），App 位于 `Sentinel EVC.app`。例如：

```bash
Sentinel-EVC/bin/sentinel-evc experiments
Sentinel-EVC/bin/sentinel-evc physics --out runs/physics-01 --render
open "Sentinel-EVC/Sentinel EVC.app"
```

## App 的运行边界

App 是 1320×900 的原生窗口，启动安装目录中的受管 Python，等待后端公布随机端口，再打开 `127.0.0.1` 工作台。启动页和错误页都在 App 内显示；退出 App 会终止子进程。后端仍执行 Host、Origin、CSRF 和 loopback 限制。

当前包是“Python 前置条件 + 受管虚拟环境”，不是包含 Python 的独立签名发行包，也尚未完成 Apple 签名与公证。应用配置记录解释器、数据目录和公开源码位置；移动整个安装目录后应重新安装，以免原生 App 保留旧的绝对路径。

## 单独构建 App

已经有可导入 `sentinel_evc` 的 Python 时，可从源码检出构建外壳：

```bash
python tools/native_app/build_app.py \
  --out outputs \
  --python .venv/bin/python \
  --data-dir runs/workbench
```

构建需要 macOS 和 Apple Command Line Tools。前端资源随 Python 包离线提供，不使用 CDN 或前端构建工具。
