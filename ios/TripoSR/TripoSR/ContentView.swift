import SwiftUI

struct ContentView: View {
    @EnvironmentObject var appState: AppState

    var body: some View {
        ZStack {
            switch appState.flow {
            case .home:
                HomeView()

            case .picking:
                // PHPickerView / camera sheet is presented from HomeView
                // We never actually land on .picking as a full-screen destination;
                // it's handled by the sheet in HomeView.
                HomeView()

            case .processing:
                ProcessingView()

            case .preview(let url):
                MeshPreviewView(meshURL: url)

            case .error(let message):
                ErrorView(message: message)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: appState.flow)
    }
}
