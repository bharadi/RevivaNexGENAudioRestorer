fn main() {
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("assets/app_icon.ico");
        res.set("ProductName", "Reviva NexGEN Audio Restorer");
        res.set("FileDescription", "Reviva NexGEN Audio Restorer v1.0.0");
        res.set("ProductVersion", "1.0.0");
        res.set("FileVersion", "1.0.0");
        if let Err(e) = res.compile() {
            println!("cargo:warning=Failed to embed Windows icon resource: {e}");
        }
    }
}
