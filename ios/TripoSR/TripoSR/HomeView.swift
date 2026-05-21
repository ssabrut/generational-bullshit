import SwiftUI
import PhotosUI

struct HomeView: View {
    @EnvironmentObject var appState: AppState

    @State private var showSourceSheet    = false
    @State private var showPhotoPicker    = false
    @State private var showCameraPicker   = false
    @State private var showSettings       = false
    @State private var showHistory        = false
    @State private var photoPickerItem: PhotosPickerItem? = nil

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                heroSection
                Spacer()
                scanButton
                    .padding(.horizontal, 32)
                    .padding(.bottom, 48)
            }
            .background(backgroundGradient)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showHistory = true
                    } label: {
                        Image(systemName: "clock.arrow.circlepath")
                            .font(.title3)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showSettings = true
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.title3)
                    }
                }
            }
        }
        // Source selection sheet (camera vs library)
        .confirmationDialog("Choose Image Source", isPresented: $showSourceSheet, titleVisibility: .visible) {
            Button("Camera") { showCameraPicker = true }
            Button("Photo Library") { showPhotoPicker = true }
            Button("Cancel", role: .cancel) {}
        }
        // PHPicker (iOS 16+ inline picker)
        .photosPicker(
            isPresented: $showPhotoPicker,
            selection: $photoPickerItem,
            matching: .images,
            photoLibrary: .shared()
        )
        .onChange(of: photoPickerItem) { _, newItem in
            guard let newItem else { return }
            Task {
                if let data = try? await newItem.loadTransferable(type: Data.self),
                   let image = UIImage(data: data) {
                    appState.imagePicked(image)
                }
                photoPickerItem = nil
            }
        }
        // Camera
        .fullScreenCover(isPresented: $showCameraPicker) {
            CameraPickerView { image in
                showCameraPicker = false
                if let image { appState.imagePicked(image) }
            }
            .ignoresSafeArea()
        }
        // Settings
        .sheet(isPresented: $showSettings) {
            SettingsView()
        }
        // History
        .sheet(isPresented: $showHistory) {
            MeshHistoryView()
        }
    }

    // MARK: Sub-views

    private var heroSection: some View {
        VStack(spacing: 20) {
            Spacer(minLength: 60)

            Image(systemName: "cube.transparent")
                .font(.system(size: 96, weight: .thin))
                .foregroundStyle(.white.opacity(0.9))
                .symbolEffect(.pulse, options: .repeating)

            Text("TripoSR")
                .font(.system(size: 42, weight: .bold, design: .rounded))
                .foregroundStyle(.white)

            Text("Photo → 3D Mesh")
                .font(.title3)
                .foregroundStyle(.white.opacity(0.75))

            Spacer(minLength: 40)
        }
    }

    private var scanButton: some View {
        Button {
            showSourceSheet = true
        } label: {
            Label("Scan Object", systemImage: "camera.viewfinder")
                .font(.title3.bold())
                .frame(maxWidth: .infinity)
                .padding(.vertical, 18)
                .background(.white)
                .foregroundStyle(Color.accentColor)
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                .shadow(color: .black.opacity(0.15), radius: 12, y: 6)
        }
    }

    private var backgroundGradient: some View {
        LinearGradient(
            colors: [Color(hue: 0.62, saturation: 0.72, brightness: 0.45),
                     Color(hue: 0.55, saturation: 0.65, brightness: 0.3)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }
}
