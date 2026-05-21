import SwiftUI

struct ErrorView: View {
    let message: String
    @EnvironmentObject var appState: AppState

    var body: some View {
        ZStack {
            LinearGradient(
                colors: [Color(hue: 0.62, saturation: 0.72, brightness: 0.35),
                         Color(hue: 0.55, saturation: 0.65, brightness: 0.2)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
            .ignoresSafeArea()

            VStack(spacing: 32) {
                Spacer()

                Image(systemName: "exclamationmark.triangle.fill")
                    .font(.system(size: 64))
                    .foregroundStyle(.yellow)

                VStack(spacing: 12) {
                    Text("Something went wrong")
                        .font(.title2.bold())
                        .foregroundStyle(.white)

                    Text(message)
                        .font(.body)
                        .foregroundStyle(.white.opacity(0.8))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 32)
                }

                Spacer()

                VStack(spacing: 14) {
                    // Retry with the same image if available
                    if appState.selectedImage != nil {
                        Button {
                            appState.retryWithCurrentImage()
                        } label: {
                            Label("Try Again", systemImage: "arrow.clockwise")
                                .font(.headline)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(.white.opacity(0.2))
                                .foregroundStyle(.white)
                                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                        }
                    }

                    Button {
                        appState.dismissError()
                    } label: {
                        Label("Back to Home", systemImage: "house")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(.white)
                            .foregroundStyle(Color.accentColor)
                            .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    }
                }
                .padding(.horizontal, 32)
                .padding(.bottom, 48)
            }
        }
    }
}
