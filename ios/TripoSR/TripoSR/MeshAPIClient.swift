import Foundation
import UIKit

// MARK: - Errors

enum APIError: LocalizedError {
    case invalidURL
    case httpError(Int, String)
    case decodingError(String)
    case noData

    var errorDescription: String? {
        switch self {
        case .invalidURL:             return "Invalid server URL."
        case .httpError(let c, let m): return "Server error \(c): \(m)"
        case .decodingError(let m):   return "Response error: \(m)"
        case .noData:                 return "Server returned no data."
        }
    }
}

// MARK: - Progress wrapper

/// Wraps URLSession delegate callbacks into an async stream of upload-fraction [0…1].
final class UploadProgressDelegate: NSObject, URLSessionTaskDelegate {
    var onProgress: ((Double) -> Void)?

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didSendBodyData bytesSent: Int64,
        totalBytesSent: Int64,
        totalBytesExpectedToSend: Int64
    ) {
        guard totalBytesExpectedToSend > 0 else { return }
        let fraction = Double(totalBytesSent) / Double(totalBytesExpectedToSend)
        onProgress?(fraction)
    }
}

// MARK: - Client

struct MeshAPIClient {

    let serverURL: URL
    let format: MeshFormat

    // MARK: Health check

    func checkHealth() async throws {
        let url = serverURL.appendingPathComponent("health")
        let (_, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw APIError.httpError(
                (response as? HTTPURLResponse)?.statusCode ?? 0,
                "Health check failed"
            )
        }
    }

    // MARK: Generate mesh

    /// Upload `image`, report upload progress via `onProgress` (0…1),
    /// then wait for the server to finish and return the raw GLB/OBJ bytes.
    func generateMesh(
        image: UIImage,
        onProgress: @escaping (UploadPhase, Double) -> Void
    ) async throws -> Data {
        let imageData = try compressImage(image)

        // Build multipart body
        let boundary = "Boundary-\(UUID().uuidString)"
        let body = buildMultipart(boundary: boundary, imageData: imageData, format: format)

        var request = URLRequest(url: serverURL.appendingPathComponent("mesh"))
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        // Inform server of expected size so it can stream efficiently
        request.setValue("\(body.count)", forHTTPHeaderField: "Content-Length")
        // Mesh extraction can take up to ~120 s on CPU
        request.timeoutInterval = 180

        // Upload with progress
        onProgress(.uploading, 0)
        let progressDelegate = UploadProgressDelegate()
        progressDelegate.onProgress = { fraction in
            onProgress(.uploading, fraction)
        }
        let session = URLSession(
            configuration: .default,
            delegate: progressDelegate,
            delegateQueue: nil
        )
        onProgress(.processing, 0)

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await session.upload(for: request, from: body)
        } catch {
            throw error
        }

        guard let http = response as? HTTPURLResponse else {
            throw APIError.noData
        }
        guard http.statusCode == 200 else {
            let message = String(data: data, encoding: .utf8) ?? "unknown"
            throw APIError.httpError(http.statusCode, message)
        }
        guard !data.isEmpty else {
            throw APIError.noData
        }

        onProgress(.done, 1)
        return data
    }

    // MARK: Private helpers

    private func compressImage(_ image: UIImage) throws -> Data {
        // Downscale to 1024 max dimension before sending — the server will
        // resize to 512 anyway, so sending a 12 MP photo just wastes bandwidth.
        let maxDim: CGFloat = 1024
        let size = image.size
        let scale = min(maxDim / size.width, maxDim / size.height, 1)
        let targetSize = CGSize(width: size.width * scale, height: size.height * scale)

        let renderer = UIGraphicsImageRenderer(size: targetSize)
        let resized = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }

        guard let data = resized.jpegData(compressionQuality: 0.88) else {
            throw APIError.decodingError("Could not encode image as JPEG")
        }
        return data
    }

    private func buildMultipart(boundary: String, imageData: Data, format: MeshFormat) -> Data {
        var body = Data()
        let crlf = "\r\n"

        func append(_ string: String) {
            body.append(Data(string.utf8))
        }

        // -- image field
        append("--\(boundary)\(crlf)")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"photo.jpg\"\(crlf)")
        append("Content-Type: image/jpeg\(crlf)\(crlf)")
        body.append(imageData)
        append(crlf)

        // -- format field
        append("--\(boundary)\(crlf)")
        append("Content-Disposition: form-data; name=\"format\"\(crlf)\(crlf)")
        append(format.rawValue)
        append(crlf)

        append("--\(boundary)--\(crlf)")
        return body
    }
}

// MARK: - Supporting types

enum MeshFormat: String, CaseIterable, Identifiable {
    case glb, obj
    var id: String { rawValue }
    var displayName: String { rawValue.uppercased() }
    var fileExtension: String { rawValue }
    var mimeType: String {
        switch self {
        case .glb: return "model/gltf-binary"
        case .obj: return "text/plain"
        }
    }
}

enum UploadPhase: String {
    case uploading  = "Uploading image…"
    case processing = "Generating mesh on server…"
    case done       = "Done"
}
