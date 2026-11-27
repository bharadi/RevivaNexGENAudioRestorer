#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod audio;
mod config;
mod ml;
mod ui;

use ui::NexGenDesktopApp;

fn load_app_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!("../assets/app_icon.png");
    let img = image::load_from_memory(bytes).ok()?.into_rgba8();
    let (width, height) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width,
        height,
    })
}

fn main() -> eframe::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size([1180.0, 780.0])
        .with_min_inner_size([920.0, 620.0])
        .with_title("Reviva NexGEN Audio Restorer");
    if let Some(icon) = load_app_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "Reviva NexGEN Audio Restorer",
        native_options,
        Box::new(|cc| Ok(Box::new(NexGenDesktopApp::new(cc)))),
    )
}
