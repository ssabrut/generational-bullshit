import SwiftUI
import Combine

// MARK: - Flow state

enum AppFlow: Equatable {
    case home
    case picking
    case processing
    case preview(URL)          // local file URL of the downloaded mesh
    case error(String)
}

// MARK: - AppState

@MainActor
final class AppState: ObservableObject {

    // Persisted settings
    @AppStorage("serverURL")  var serverURLString: String = "http://localhost:8000"
    @AppStorage("meshFormat") var meshFormatRaw:   String = MeshFormat.glb.rawValue

    var meshFormat: MeshFormat {
        get { MeshFormat(rawValue: meshFormatRaw) ?? .glb }
        set { meshFormatRaw = newValue.rawValue }
    }

    // Flow
    @Published var flow: AppFlow = .home

    // Processing state (shown in ProcessingView)
    @Published var uploadPhase: UploadPhase = .uploading
    @Published var uploadProgress: Double   = 0        // 0…1 for the upload fraction
    @Published var selectedImage: UIImage?

    // Result meshes saved to the Documents/Meshes folder
    @Published var savedMeshURLs: [URL] = []

    // MARK: Init

    init() {
        refreshSavedMeshes()
    }

    // MARK: Actions

    func startPicking() {
        flow = .picking
    }

    func cancelPicking() {
        flow = .home
    }

    func imagePicked(_ image: UIImage) {
        selectedImage = image
        flow = .processing
        Task { await generateMesh(from: image) }
    }

    func retryWithCurrentImage() {
        guard let image = selectedImage else { flow = .home; return }
        flow = .processing
        Task { await generateMesh(from: image) }
    }

    func dismissError() {
        flow = .home
    }

    func dismissPreview() {
        flow = .home
    }

    // MARK: Private

    private func generateMesh(from image: UIImage) async {
        uploadPhase    = .uploading
        uploadProgress = 0

        guard let url = URL(string: serverURLString), !serverURLString.isEmpty else {
            flow = .error("No server URL configured. Tap ⚙️ to set one.")
            return
        }

        let client = MeshAPIClient(serverURL: url, format: meshFormat)

        do {
            let meshData = try await client.generateMesh(image: image) { [weak self] phase, progress in
                Task { @MainActor [weak self] in
                    self?.uploadPhase    = phase
                    self?.uploadProgress = progress
                }
            }

            let savedURL = try saveMesh(meshData, format: meshFormat)
            refreshSavedMeshes()
            flow = .preview(savedURL)

        } catch {
            flow = .error(error.localizedDescription)
        }
    }

    private func saveMesh(_ data: Data, format: MeshFormat) throws -> URL {
        let dir = try meshesDirectory()
        let name = "mesh_\(Int(Date().timeIntervalSince1970)).\(format.fileExtension)"
        let url  = dir.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    private func meshesDirectory() throws -> URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir  = docs.appendingPathComponent("Meshes", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func refreshSavedMeshes() {
        guard let dir = try? meshesDirectory() else { return }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir,
            includingPropertiesForKeys: [.creationDateKey],
            options: .skipsHiddenFiles
        )) ?? []
        savedMeshURLs = files
            .filter { ["glb", "obj"].contains($0.pathExtension) }
            .sorted {
                let d0 = (try? $0.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                let d1 = (try? $1.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
                return d0 > d1
            }
    }

    func deleteMesh(at url: URL) {
        try? FileManager.default.removeItem(at: url)
        refreshSavedMeshes()
    }
}
