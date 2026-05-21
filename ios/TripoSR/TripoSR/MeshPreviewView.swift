import SwiftUI
import SceneKit
import QuickLook

struct MeshPreviewView: View {
    let meshURL: URL

    @EnvironmentObject var appState: AppState
    @State private var showShareSheet  = false
    @State private var showQuickLook   = false
    @State private var sceneError: String? = nil

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            VStack(spacing: 0) {
                // SceneKit viewport
                if let errorMsg = sceneError {
                    sceneErrorPlaceholder(errorMsg)
                } else {
                    SceneKitView(url: meshURL, onError: { sceneError = $0 })
                }

                bottomBar
            }
        }
        .sheet(isPresented: $showShareSheet) {
            ShareSheet(items: [meshURL])
        }
        .fullScreenCover(isPresented: $showQuickLook) {
            QuickLookPreview(url: meshURL)
                .ignoresSafeArea()
        }
        .navigationBarBackButtonHidden(true)
    }

    // MARK: Bottom bar

    private var bottomBar: some View {
        HStack(spacing: 0) {
            // Back to home
            Button {
                appState.dismissPreview()
            } label: {
                Label("New Scan", systemImage: "camera.viewfinder")
                    .font(.subheadline.bold())
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
            }

            Divider().frame(height: 30)

            // AR / Quick Look
            Button {
                showQuickLook = true
            } label: {
                Label("AR View", systemImage: "arkit")
                    .font(.subheadline.bold())
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
            }

            Divider().frame(height: 30)

            // Share / export
            Button {
                showShareSheet = true
            } label: {
                Label("Export", systemImage: "square.and.arrow.up")
                    .font(.subheadline.bold())
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
            }
        }
        .foregroundStyle(.primary)
        .background(.ultraThinMaterial)
    }

    private func sceneErrorPlaceholder(_ message: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.yellow)
            Text("Could not render mesh")
                .font(.headline)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Button("Open with Quick Look") { showQuickLook = true }
                .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

// MARK: - SceneKit UIViewRepresentable

/// Full-featured SceneKit view with:
///   - auto-lit scene (two directional lights + ambient)
///   - orbit via single-finger drag
///   - pinch to zoom
///   - double-tap to reset camera
struct SceneKitView: UIViewRepresentable {
    let url: URL
    var onError: (String) -> Void

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = UIColor.black
        view.autoenablesDefaultLighting = false
        view.allowsCameraControl = false   // we do manual gestures below
        view.antialiasingMode = .multisampling4X

        // Gesture recognisers
        let pan = UIPanGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePan(_:)))
        let pinch = UIPinchGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handlePinch(_:)))
        let doubleTap = UITapGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleDoubleTap(_:)))
        doubleTap.numberOfTapsRequired = 2
        view.addGestureRecognizer(pan)
        view.addGestureRecognizer(pinch)
        view.addGestureRecognizer(doubleTap)
        context.coordinator.scnView = view

        loadScene(into: view)
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    // MARK: Scene loading

    private func loadScene(into view: SCNView) {
        do {
            let scene = try SCNScene(url: url, options: [
                .checkConsistency: false,
                .flattenScene: false,
            ])

            // Build a pivot node so we can orbit around the mesh centre
            let pivot = SCNNode()
            pivot.name = "pivot"
            for child in scene.rootNode.childNodes {
                pivot.addChildNode(child)
            }
            scene.rootNode.addChildNode(pivot)

            // Centre and normalise scale
            let (minVec, maxVec) = pivot.boundingBox
            let centre = SCNVector3(
                (minVec.x + maxVec.x) / 2,
                (minVec.y + maxVec.y) / 2,
                (minVec.z + maxVec.z) / 2
            )
            let extent = max(
                maxVec.x - minVec.x,
                maxVec.y - minVec.y,
                maxVec.z - minVec.z
            )
            // Normalise so the mesh fits within a unit cube
            let scale = extent > 0 ? Float(1.0 / extent) : 1
            pivot.pivot = SCNMatrix4MakeTranslation(centre.x, centre.y, centre.z)
            pivot.scale = SCNVector3(scale, scale, scale)

            // Lights
            addLights(to: scene)

            // Camera
            let cameraNode = SCNNode()
            cameraNode.name = "camera"
            cameraNode.camera = SCNCamera()
            cameraNode.camera?.zNear = 0.01
            cameraNode.camera?.zFar = 100
            cameraNode.position = SCNVector3(0, 0, 2.5)
            scene.rootNode.addChildNode(cameraNode)

            view.scene = scene
            view.pointOfView = cameraNode
        } catch {
            onError(error.localizedDescription)
        }
    }

    private func addLights(to scene: SCNScene) {
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 400
        ambient.light?.color = UIColor.white
        scene.rootNode.addChildNode(ambient)

        func directional(euler: SCNVector3, intensity: CGFloat) -> SCNNode {
            let n = SCNNode()
            n.light = SCNLight()
            n.light?.type = .directional
            n.light?.intensity = intensity
            n.light?.color = UIColor.white
            n.eulerAngles = euler
            return n
        }
        scene.rootNode.addChildNode(directional(euler: SCNVector3(-0.7, 0.5, 0), intensity: 800))
        scene.rootNode.addChildNode(directional(euler: SCNVector3(0.5, -0.8, 0.3), intensity: 500))
    }

    // MARK: - Coordinator (gesture handling)

    final class Coordinator: NSObject {
        weak var scnView: SCNView?

        // Orbit state
        private var lastPanLocation: CGPoint = .zero
        private var rotationX: Float = -0.3   // initial slight top-down angle
        private var rotationY: Float = 0

        // Zoom state
        private var lastPinchScale: CGFloat = 1
        private var cameraDistance: Float = 2.5

        private var pivotNode: SCNNode? { scnView?.scene?.rootNode.childNode(withName: "pivot", recursively: false) }
        private var cameraNode: SCNNode? { scnView?.scene?.rootNode.childNode(withName: "camera", recursively: false) }

        @objc func handlePan(_ gr: UIPanGestureRecognizer) {
            guard let view = scnView else { return }
            let loc = gr.location(in: view)

            if gr.state == .began {
                lastPanLocation = loc
                return
            }

            let dx = Float(loc.x - lastPanLocation.x)
            let dy = Float(loc.y - lastPanLocation.y)
            lastPanLocation = loc

            let sensitivity: Float = 0.005
            rotationY += dx * sensitivity
            rotationX += dy * sensitivity
            rotationX = max(-Float.pi / 2, min(Float.pi / 2, rotationX))

            applyRotation()
        }

        @objc func handlePinch(_ gr: UIPinchGestureRecognizer) {
            if gr.state == .began { lastPinchScale = gr.scale; return }
            let delta = Float(gr.scale / lastPinchScale)
            lastPinchScale = gr.scale
            cameraDistance = max(0.5, min(8.0, cameraDistance / delta))
            cameraNode?.position = SCNVector3(0, 0, cameraDistance)
        }

        @objc func handleDoubleTap(_ gr: UITapGestureRecognizer) {
            rotationX = -0.3; rotationY = 0; cameraDistance = 2.5
            applyRotation()
            cameraNode?.position = SCNVector3(0, 0, cameraDistance)
        }

        private func applyRotation() {
            pivotNode?.eulerAngles = SCNVector3(rotationX, rotationY, 0)
        }
    }
}

// MARK: - Quick Look

struct QuickLookPreview: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> QLPreviewController {
        let controller = QLPreviewController()
        controller.dataSource = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: QLPreviewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(url: url) }

    final class Coordinator: NSObject, QLPreviewControllerDataSource {
        let url: URL
        init(url: URL) { self.url = url }
        func numberOfPreviewItems(in controller: QLPreviewController) -> Int { 1 }
        func previewController(_ controller: QLPreviewController, previewItemAt index: Int) -> QLPreviewItem {
            url as QLPreviewItem
        }
    }
}

// MARK: - Share sheet

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}
