import AppKit
import Darwin
import Foundation
import WebKit

private struct AppConfig: Decodable {
    let schema_version: String
    let python: String
    let data_dir: String
    let source_dir: String
    let bind: String
}

@main
final class SentinelApp: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKDownloadDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var backend: Process?
    private var outputBuffer = Data()
    private var terminating = false
    private var backendPort: Int?

    static func main() {
        let app = NSApplication.shared
        let delegate = SentinelApp()
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        configureMainMenu()
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1320, height: 900),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered, defer: false
        )
        window.title = "Sentinel EVC"
        window.titlebarAppearsTransparent = true
        window.minSize = NSSize(width: 960, height: 680)
        window.contentView = webView
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        showStatus(title: "正在启动实验工作台", detail: "正在创建仅本机可访问的会话…", failed: false)
        startBackend()
    }

    private func configureMainMenu() {
        let mainMenu = NSMenu()

        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu(title: "Sentinel EVC")
        let about = NSMenuItem(
            title: "关于 Sentinel EVC",
            action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
            keyEquivalent: ""
        )
        about.target = NSApp
        appMenu.addItem(about)
        appMenu.addItem(.separator())
        let quit = NSMenuItem(
            title: "退出 Sentinel EVC",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        )
        quit.target = NSApp
        appMenu.addItem(quit)
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let editMenuItem = NSMenuItem()
        let editMenu = NSMenu(title: "编辑")
        editMenu.addItem(NSMenuItem(title: "复制", action: #selector(NSText.copy(_:)), keyEquivalent: "c"))
        editMenu.addItem(NSMenuItem(title: "粘贴", action: #selector(NSText.paste(_:)), keyEquivalent: "v"))
        editMenu.addItem(NSMenuItem(title: "全选", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"))
        editMenuItem.submenu = editMenu
        mainMenu.addItem(editMenuItem)

        NSApp.mainMenu = mainMenu
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        terminating = true
        stopBackend()
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else {
            decisionHandler(.cancel)
            return
        }
        let localPage = url.scheme == "about"
        let backendPage = url.scheme == "http" && url.host == "127.0.0.1"
            && backendPort != nil && url.port == backendPort
        if backendPage && action.shouldPerformDownload {
            decisionHandler(.download)
        } else {
            decisionHandler(localPage || backendPage ? .allow : .cancel)
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        guard let url = response.response.url else {
            decisionHandler(.cancel)
            return
        }
        let backendResponse = url.scheme == "http" && url.host == "127.0.0.1"
            && backendPort != nil && url.port == backendPort
        guard backendResponse else {
            decisionHandler(.cancel)
            return
        }
        decisionHandler(response.canShowMIMEType ? .allow : .download)
    }

    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction,
                 didBecome download: WKDownload) {
        download.delegate = self
    }

    func webView(_ webView: WKWebView, navigationResponse: WKNavigationResponse,
                 didBecome download: WKDownload) {
        download.delegate = self
    }

    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse,
                  suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = URL(fileURLWithPath: suggestedFilename).lastPathComponent
        panel.beginSheetModal(for: window) { result in
            completionHandler(result == .OK ? panel.url : nil)
        }
    }

    func downloadDidFinish(_ download: WKDownload) {
        showDownloadNotice(title: "证据已保存", detail: "下载已完成。")
    }

    func download(_ download: WKDownload, didFailWithError error: Error,
                  resumeData: Data?) {
        showDownloadNotice(title: "保存失败", detail: error.localizedDescription)
    }

    private func startBackend() {
        guard let configURL = Bundle.main.url(forResource: "app-config", withExtension: "json"),
              let data = try? Data(contentsOf: configURL),
              let config = try? JSONDecoder().decode(AppConfig.self, from: data),
              config.schema_version == "native-app-v1", config.bind == "127.0.0.1" else {
            showStatus(title: "应用配置不可用", detail: "请从安装器重新构建 Sentinel EVC.app。", failed: true)
            return
        }
        let task = Process()
        task.executableURL = URL(fileURLWithPath: config.python)
        task.arguments = ["-m", "sentinel_evc", "serve", "--port", "0", "--data-dir", config.data_dir]
        var environment = ProcessInfo.processInfo.environment
        if !config.source_dir.isEmpty {
            let source = URL(fileURLWithPath: config.source_dir).appendingPathComponent("src").path
            environment["PYTHONPATH"] = source + (environment["PYTHONPATH"].map { ":" + $0 } ?? "")
        }
        task.environment = environment
        let pipe = Pipe()
        task.standardOutput = pipe
        task.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData
            guard !bytes.isEmpty else { return }
            DispatchQueue.main.async { self?.consume(bytes) }
        }
        task.terminationHandler = { [weak self] process in
            DispatchQueue.main.async {
                guard let self = self, !self.terminating else { return }
                self.showStatus(title: "本地服务已停止", detail: "退出状态 \(process.terminationStatus)。请检查 Python 安装后重试。", failed: true)
            }
        }
        do {
            try task.run()
            backend = task
        } catch {
            showStatus(title: "无法启动本地服务", detail: error.localizedDescription, failed: true)
        }
    }

    private func consume(_ data: Data) {
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 10) {
            let lineData = outputBuffer.prefix(upTo: newline)
            outputBuffer.removeSubrange(...newline)
            guard let line = String(data: lineData, encoding: .utf8),
                  let range = line.range(of: "http://127.0.0.1:") else { continue }
            let raw = line[range.lowerBound...].trimmingCharacters(in: .whitespacesAndNewlines)
            if let url = URL(string: raw), url.host == "127.0.0.1" {
                backendPort = url.port
                webView.load(URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 15))
            }
        }
    }

    private func stopBackend() {
        guard let task = backend, task.isRunning else { return }
        task.terminate()
        // The server closes active workers and finalizes evidence on SIGTERM.
        // Keep the application alive long enough for that cleanup to finish.
        let deadline = Date().addingTimeInterval(8.0)
        while task.isRunning && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.05))
        }
        if task.isRunning {
            kill(task.processIdentifier, SIGKILL)
        }
        task.waitUntilExit()
    }

    private func showStatus(title: String, detail: String, failed: Bool) {
        let accent = failed ? "#b44848" : "#39a8a0"
        let html = """
        <!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
        <style>
        :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 20% 0%,#dcebea 0,transparent 38%),#f4f2ed;color:#17212b;font:15px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}.card{width:min(620px,calc(100vw - 64px));padding:42px;border:1px solid #d9dcd9;border-radius:20px;background:rgba(255,255,255,.88);box-shadow:0 24px 70px rgba(22,37,46,.11)}.mark{width:48px;height:48px;border-radius:14px;background:\(accent);box-shadow:inset 0 0 0 12px rgba(255,255,255,.78);margin-bottom:28px}h1{margin:0 0 12px;font-size:29px;letter-spacing:-.02em}p{margin:0;color:#64717a;line-height:1.65}.scope{margin-top:28px;padding-top:20px;border-top:1px solid #e6e7e3;color:#7c858a;font-size:12px;letter-spacing:.04em;text-transform:uppercase}</style>
        <main class="card"><div class="mark"></div><h1>\(escape(title))</h1><p>\(escape(detail))</p><div class="scope">Loopback session · signed evidence</div></main></html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }

    private func showDownloadNotice(title: String, detail: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = detail
        alert.addButton(withTitle: "好")
        alert.beginSheetModal(for: window)
    }

    private func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
    }
}
