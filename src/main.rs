#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu_thingi::run;

fn main() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
        } else {
            tracing_subscriber::fmt::init();
        }
    }
    pollster::block_on(run());
}
