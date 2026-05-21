import SwiftUI
import SceneKit

struct MeshHistoryView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var selectedURL: URL? = nil

    private let columns = [GridItem(.adaptive(minimum: 160), spacing: 16)]

    var body: some View {
        NavigationStack {
            Group {
                if appState.savedMeshURLs.isEmpty {
                    emptyState
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 16) {
                            ForEach(appState.savedMeshURLs, id: \.self) { url in
                                MeshThumbnailCard(url: url) {
                                    selectedURL = url
                                    dismiss()
                                    // Navigate to preview after dismiss completes
                                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                                        appState.flow = .preview(url)
                                    }
                                }
                                .contextMenu {
                                    Button(role: .destructive) {
                                        appState.deleteMesh(at: url)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                    Button {
                                        selectedURL = url
                                    } label: {
                                        Label("Share", systemImage: "square.and.arrow.up")
                                    }
                                }
                            }
                        }
                        .padding(16)
                    }
                }
            }
            .navigationTitle("Mesh History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .destructiveAction) {
                    if !appState.savedMeshURLs.isEmpty {
                        Button(role: .destructive) {
                            appState.savedMeshURLs.forEach { appState.deleteMesh(at: $0) }
                        } label: {
                            Image(systemName: "trash")
                        }
                    }
                }
            }
            .sheet(item: $selectedURL) { url in
                ShareSheet(items: [url])
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "clock.badge.xmark")
                .font(.system(size: 56))
                .foregroundStyle(.secondary)
            Text("No meshes yet")
                .font(.title3.bold())
            Text("Scan an object to generate your first 3D mesh.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

// MARK: URL + Identifiable

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

// MARK: Thumbnail card

private struct MeshThumbnailCard: View {
    let url: URL
    let onTap: () -> Void

    var body: some View {
        Button(action: onTap) {
            VStack(alignment: .leading, spacing: 8) {
                // Mini SceneKit thumbnail
                MiniSceneView(url: url)
                    .frame(height: 140)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(url.deletingPathExtension().lastPathComponent)
                        .font(.caption.bold())
                        .lineLimit(1)
                        .truncationMode(.middle)

                    Text(url.pathExtension.uppercased())
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary, in: Capsule())
                }
                .padding(.horizontal, 4)
            }
        }
        .buttonStyle(.plain)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

// MARK: Mini SceneKit thumbnail (static, no gestures)

private struct MiniSceneView: UIViewRepresentable {
    let url: URL

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = UIColor.systemGray6
        view.autoenablesDefaultLighting = true
        view.isUserInteractionEnabled = false

        if let scene = try? SCNScene(url: url, options: [.checkConsistency: false]) {
            let pivot = SCNNode()
            for child in scene.rootNode.childNodes { pivot.addChildNode(child) }
            scene.rootNode.addChildNode(pivot)

            let (min, max) = pivot.boundingBox
            let centre = SCNVector3((min.x+max.x)/2, (min.y+max.y)/2, (min.z+max.z)/2)
            let extent = Swift.max(max.x-min.x, max.y-min.y, max.z-min.z)
            let scale = extent > 0 ? Float(1.0/extent) : 1
            pivot.pivot = SCNMatrix4MakeTranslation(centre.x, centre.y, centre.z)
            pivot.scale = SCNVector3(scale, scale, scale)
            pivot.eulerAngles = SCNVector3(-0.4, 0.5, 0)

            let cam = SCNNode(); cam.camera = SCNCamera()
            cam.position = SCNVector3(0, 0, 2.2)
            scene.rootNode.addChildNode(cam)
            view.scene = scene
            view.pointOfView = cam
        }
        return view
    }

    func updateUIView(_ uiView: SCNView, context: Context) {}
}
