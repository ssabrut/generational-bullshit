import SwiftUI

struct ProcessingView: View {
    @EnvironmentObject var appState: AppState

    // Drive a looping spinner angle
    @State private var spinAngle: Double = 0

    var body: some View {
        ZStack {
            backgroundGradient

            VStack(spacing: 40) {
                // Thumbnail of the photo being processed
                if let image = appState.selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFill()
                        .frame(width: 200, height: 200)
                        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                        .overlay(
                            RoundedRectangle(cornerRadius: 20, style: .continuous)
                                .stroke(.white.opacity(0.3), lineWidth: 1)
                        )
                        .shadow(color: .black.opacity(0.3), radius: 16, y: 8)
                }

                VStack(spacing: 24) {
                    // Animated mesh icon
                    ZStack {
                        Circle()
                            .stroke(.white.opacity(0.15), lineWidth: 3)
                            .frame(width: 80, height: 80)

                        Image(systemName: "cube.transparent")
                            .font(.system(size: 36))
                            .foregroundStyle(.white)
                            .rotationEffect(.degrees(spinAngle))
                            .onAppear {
                                withAnimation(.linear(duration: 3).repeatForever(autoreverses: false)) {
                                    spinAngle = 360
                                }
                            }
                    }

                    // Phase label
                    Text(appState.uploadPhase.rawValue)
                        .font(.headline)
                        .foregroundStyle(.white)
                        .animation(.default, value: appState.uploadPhase)

                    // Progress bar (only meaningful during upload)
                    if appState.uploadPhase == .uploading {
                        VStack(spacing: 8) {
                            ProgressView(value: appState.uploadProgress)
                                .progressViewStyle(LinearProgressViewStyle(tint: .white))
                                .frame(maxWidth: 260)

                            Text("\(Int(appState.uploadProgress * 100))%")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.7))
                                .monospacedDigit()
                        }
                    } else {
                        // Indeterminate spinner while server processes
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(1.2)
                    }
                }
                .padding(36)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 28, style: .continuous))

                // Step indicators
                StepIndicatorRow()
                    .padding(.horizontal, 32)
            }
            .padding(.vertical, 60)
        }
    }

    private var backgroundGradient: some View {
        LinearGradient(
            colors: [Color(hue: 0.62, saturation: 0.72, brightness: 0.35),
                     Color(hue: 0.55, saturation: 0.65, brightness: 0.2)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
}

// MARK: - Step indicator

private struct MeshStep: Identifiable {
    let id: Int
    let phase: UploadPhase
    let icon: String
    let label: String
}

private struct StepIndicatorRow: View {
    @EnvironmentObject var appState: AppState

    private let steps: [MeshStep] = [
        MeshStep(id: 0, phase: .uploading,  icon: "arrow.up.circle", label: "Upload"),
        MeshStep(id: 1, phase: .processing, icon: "cpu",              label: "Generate"),
        MeshStep(id: 2, phase: .done,       icon: "checkmark.circle", label: "Done"),
    ]

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(steps.enumerated()), id: \.element.id) { index, step in
                let isActive   = appState.uploadPhase == step.phase
                let isDone     = isDoneOrPast(step.phase)
                let icon       = step.icon
                let label      = step.label

                VStack(spacing: 6) {
                    ZStack {
                        Circle()
                            .fill(isDone ? Color.white : (isActive ? Color.white.opacity(0.4) : Color.white.opacity(0.1)))
                            .frame(width: 40, height: 40)

                        Image(systemName: icon)
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundStyle(isDone ? Color.accentColor : .white)
                    }

                    Text(label)
                        .font(.caption2.bold())
                        .foregroundStyle(isActive || isDone ? .white : .white.opacity(0.4))
                }
                .frame(maxWidth: .infinity)

                if index < (steps.count - 1) {
                    Rectangle()
                        .fill(.white.opacity(0.3))
                        .frame(height: 2)
                        .frame(maxWidth: 40)
                }
            }
        }
    }

    private func isDoneOrPast(_ phase: UploadPhase) -> Bool {
        let order: [UploadPhase] = [.uploading, .processing, .done]
        guard let current = order.firstIndex(of: appState.uploadPhase),
              let target  = order.firstIndex(of: phase) else { return false }
        return current > target
    }
}
