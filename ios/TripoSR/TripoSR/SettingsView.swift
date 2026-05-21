import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var appState: AppState
    @Environment(\.dismiss) private var dismiss

    @State private var urlDraft: String = ""
    @State private var isCheckingHealth = false
    @State private var healthStatus: HealthStatus? = nil

    enum HealthStatus {
        case ok, failed(String)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextField("http://192.168.1.10:8000", text: $urlDraft)
                        .keyboardType(.URL)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)

                    Button {
                        Task { await checkHealth() }
                    } label: {
                        HStack {
                            if isCheckingHealth {
                                ProgressView().scaleEffect(0.8)
                            } else {
                                Image(systemName: "wifi")
                            }
                            Text("Test Connection")
                        }
                    }
                    .disabled(isCheckingHealth || urlDraft.isEmpty)

                    if let status = healthStatus {
                        switch status {
                        case .ok:
                            Label("Server reachable", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        case .failed(let msg):
                            Label(msg, systemImage: "xmark.circle.fill")
                                .foregroundStyle(.red)
                                .font(.caption)
                        }
                    }
                } header: {
                    Text("Server URL")
                } footer: {
                    Text("The FastAPI server running uvicorn server.app:app. Use your Mac's LAN IP when testing on a real iPad.")
                }

                Section("Output Format") {
                    Picker("Format", selection: $appState.meshFormat) {
                        ForEach(MeshFormat.allCases) { fmt in
                            Text(fmt.displayName).tag(fmt)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                Section("About") {
                    LabeledContent("Pipeline", value: "TripoSR · Option A")
                    LabeledContent("Backend", value: "FastAPI + ONNX + PyTorch")
                    LabeledContent("Viewer", value: "SceneKit + QuickLook AR")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        appState.serverURLString = urlDraft
                        dismiss()
                    }
                }
            }
            .onAppear {
                urlDraft = appState.serverURLString
            }
        }
    }

    private func checkHealth() async {
        isCheckingHealth = true
        healthStatus = nil
        guard let url = URL(string: urlDraft) else {
            healthStatus = .failed("Invalid URL")
            isCheckingHealth = false
            return
        }
        do {
            try await MeshAPIClient(serverURL: url, format: appState.meshFormat).checkHealth()
            healthStatus = .ok
        } catch {
            healthStatus = .failed(error.localizedDescription)
        }
        isCheckingHealth = false
    }
}
