use egui::{Color32, Pos2, Rect, Rounding, Stroke, Vec2};

pub const BG_DARK: Color32 = Color32::from_rgb(10, 10, 15);
pub const PANEL_BG: Color32 = Color32::from_rgb(20, 20, 34);
pub const CARD_BG: Color32 = Color32::from_rgb(26, 26, 46);
pub const BORDER_COLOR: Color32 = Color32::from_rgb(46, 46, 72);
pub const NEON_PURPLE: Color32 = Color32::from_rgb(139, 92, 246);
pub const NEON_CYAN: Color32 = Color32::from_rgb(6, 182, 212);
pub const SUCCESS_GREEN: Color32 = Color32::from_rgb(34, 197, 94);
pub const WARN_AMBER: Color32 = Color32::from_rgb(245, 158, 11);
pub const ERROR_RED: Color32 = Color32::from_rgb(239, 68, 68);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(156, 163, 175);

pub fn apply_nexgen_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.spacing.slider_rail_height = 8.0;
    style.spacing.slider_width = 140.0;

    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = BG_DARK;
    visuals.panel_fill = PANEL_BG;
    visuals.extreme_bg_color = Color32::from_rgb(48, 52, 84);
    visuals.faint_bg_color = CARD_BG;
    visuals.slider_trailing_fill = true;

    let rail_border = Color32::from_rgb(110, 118, 175);
    visuals.widgets.noninteractive.bg_fill = CARD_BG;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER_COLOR);
    // Distinct slate-blue rail fill so slider bars stand out clearly against dark panels
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(66, 72, 114);
    visuals.widgets.inactive.bg_stroke = Stroke::new(1.2_f32, rail_border);
    visuals.widgets.inactive.fg_stroke = Stroke::new(2.0_f32, Color32::WHITE);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(84, 92, 144);
    visuals.widgets.hovered.bg_stroke = Stroke::new(1.5_f32, NEON_CYAN);
    visuals.widgets.hovered.fg_stroke = Stroke::new(2.0_f32, NEON_CYAN);
    visuals.widgets.active.bg_fill = NEON_PURPLE;
    visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
    visuals.widgets.active.fg_stroke = Stroke::new(2.0_f32, Color32::WHITE);
    visuals.selection.bg_fill = NEON_CYAN;
    visuals.selection.stroke = Stroke::new(1.5_f32, Color32::WHITE);

    style.visuals = visuals;
    ctx.set_style(style);
}

/// Draws an interactive waveform bar graph and returns `Some(normalized_seek)` if clicked.
pub fn draw_waveform(
    ui: &mut egui::Ui,
    peaks: &[f32],
    progress: f32,
    accent: Color32,
    height: f32,
    loop_range: Option<(f32, f32)>,
) -> Option<f32> {
    let desired_size = Vec2::new(ui.available_width(), height);
    let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::click());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, Rounding::same(10.0), BG_DARK);
    painter.rect_stroke(
        rect,
        Rounding::same(10.0),
        Stroke::new(1.0, BORDER_COLOR),
    );

    if peaks.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No waveform loaded",
            egui::FontId::proportional(13.0),
            TEXT_MUTED,
        );
        return None;
    }

    let inner = rect.shrink2(Vec2::new(10.0, 8.0));
    let n = peaks.len();
    let step = inner.width() / n as f32;
    let bar_w = (step * 0.65).max(1.5);
    let mid_y = inner.center().y;
    let max_half_h = inner.height() * 0.46;

    for (i, &p) in peaks.iter().enumerate() {
        let frac = i as f32 / n as f32;
        let x = inner.left() + i as f32 * step + step * 0.5;
        let h = (p.clamp(0.02, 1.0) * max_half_h).max(2.0);
        let color = if frac <= progress {
            accent
        } else {
            accent.linear_multiply(0.35)
        };
        let bar_rect = Rect::from_center_size(Pos2::new(x, mid_y), Vec2::new(bar_w, h * 2.0));
        painter.rect_filled(bar_rect, Rounding::same(1.0), color);
    }

    // Loop region markers (A-B)
    if let Some((a, b)) = loop_range {
        let ax = inner.left() + a.clamp(0.0, 1.0) * inner.width();
        let bx = inner.left() + b.clamp(0.0, 1.0) * inner.width();
        let loop_rect = Rect::from_min_max(
            Pos2::new(ax.min(bx), inner.top()),
            Pos2::new(ax.max(bx), inner.bottom()),
        );
        painter.rect_filled(loop_rect, Rounding::ZERO, SUCCESS_GREEN.linear_multiply(0.12));
        painter.line_segment(
            [Pos2::new(ax, inner.top()), Pos2::new(ax, inner.bottom())],
            Stroke::new(1.5_f32, SUCCESS_GREEN),
        );
        painter.line_segment(
            [Pos2::new(bx, inner.top()), Pos2::new(bx, inner.bottom())],
            Stroke::new(1.5_f32, WARN_AMBER),
        );
    }

    // Playhead line
    let px = inner.left() + progress.clamp(0.0, 1.0) * inner.width();
    painter.line_segment(
        [Pos2::new(px, inner.top()), Pos2::new(px, inner.bottom())],
        Stroke::new(1.5, Color32::WHITE),
    );

    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let norm = ((pos.x - inner.left()) / inner.width()).clamp(0.0, 1.0);
            return Some(norm);
        }
    }
    None
}

/// Draws a time×frequency spectrogram heatmap.
pub fn draw_spectrogram(ui: &mut egui::Ui, spec: &[Vec<f32>], height: f32) {
    let desired_size = Vec2::new(ui.available_width(), height);
    let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_filled(rect, Rounding::same(10.0), BG_DARK);
    painter.rect_stroke(
        rect,
        Rounding::same(10.0),
        Stroke::new(1.0, BORDER_COLOR),
    );

    if spec.is_empty() || spec[0].is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Spectrogram appears after audio analysis",
            egui::FontId::proportional(13.0),
            TEXT_MUTED,
        );
        return;
    }

    let inner = rect.shrink(6.0);
    let t_steps = spec.len();
    let f_bins = spec[0].len();
    let cell_w = inner.width() / t_steps as f32;
    let cell_h = inner.height() / f_bins as f32;

    for (t, col) in spec.iter().enumerate() {
        for (f, &e) in col.iter().enumerate() {
            let color = energy_to_color(e);
            let x = inner.left() + t as f32 * cell_w;
            let y = inner.bottom() - (f + 1) as f32 * cell_h;
            let cell = Rect::from_min_size(Pos2::new(x, y), Vec2::new(cell_w + 0.5, cell_h + 0.5));
            painter.rect_filled(cell, Rounding::ZERO, color);
        }
    }
}

fn energy_to_color(e: f32) -> Color32 {
    let e = e.clamp(0.0, 1.0);
    if e < 0.25 {
        let t = e / 0.25;
        lerp_color(Color32::from_rgb(10, 10, 15), Color32::from_rgb(65, 30, 130), t)
    } else if e < 0.55 {
        let t = (e - 0.25) / 0.30;
        lerp_color(Color32::from_rgb(65, 30, 130), NEON_CYAN, t)
    } else if e < 0.82 {
        let t = (e - 0.55) / 0.27;
        lerp_color(NEON_CYAN, Color32::from_rgb(251, 191, 36), t)
    } else {
        let t = (e - 0.82) / 0.18;
        lerp_color(Color32::from_rgb(251, 191, 36), Color32::WHITE, t)
    }
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * t) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * t) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * t) as u8,
    )
}
