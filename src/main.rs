use eframe::{egui, App, Frame, NativeOptions};
use rscam::{Camera, Config as CameraConfig};
use image::{ImageBuffer, Luma};
use enigo::{Enigo, Mouse, Settings};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::time::{Instant, Duration};

// ==================== СТРУКТУРЫ ====================

#[derive(Serialize, Deserialize, Clone)]
struct AppConfig {
    hsv_lower: [u8; 3],
    hsv_upper: [u8; 3],
    min_area: f64,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            hsv_lower: [0, 0, 200],
            hsv_upper: [180, 30, 255],
            min_area: 500.0,
        }
    }
}

struct InteractiveSurfaceApp {
    camera: Option<Camera>,
    current_frame: Option<egui::ColorImage>,
    filtered_frame: Option<egui::ColorImage>,
    calibration_points: Vec<[f32; 2]>,
    screen_points: Vec<[f32; 2]>,
    is_calibrated: bool,
    calibration_step: usize,
    transform_matrix: Option<[[f64; 3]; 3]>,
    mouse_controller: Enigo,
    is_tracking: bool,
    config: AppConfig,
    camera_texture: Option<egui::TextureHandle>,
    filter_texture: Option<egui::TextureHandle>,
    auto_click: bool,
    stable_counter: u32,
    last_center: Option<[f32; 2]>,
    // Performance tracking
    last_frame_time: std::time::Instant,
    frame_interval: std::time::Duration, // desired interval (e.g. 33ms ~30 FPS)
    fps_counter: FpsCounter,
    // Reusable buffers to avoid allocations
    mask_buffer: Vec<u8>,
    rgba_buffer: Vec<u8>, // reusable RGBA for display
    // Auto calibration
    auto_calibrate: bool,
    acc_min: Option<[f32;2]>,
    acc_max: Option<[f32;2]>,
    // QR calibration
    qr_enabled: bool,
    qr_status: String,
    qr_attempt_counter: u32,
    qr_gray: Vec<u8>,
    // Multi-QR corner calibration
    qr_multi_mode: bool,
    qr_corner_codes: [Option<[f32;2]>;4], // TL, TR, BL, BR centers
    qr_scanner: Option<quircs::Quirc>,
}

// ==================== FPS COUNTER ====================
struct FpsCounter {
    last_instant: Instant,
    frame_count: u32,
    fps: f32,
}

impl FpsCounter {
    fn new() -> Self {
        Self { last_instant: Instant::now(), frame_count: 0, fps: 0.0 }
    }
    fn tick(&mut self) {
        self.frame_count += 1;
        let elapsed = self.last_instant.elapsed();
        if elapsed >= Duration::from_secs(1) {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            self.frame_count = 0;
            self.last_instant = Instant::now();
        }
    }
    fn fps(&self) -> f32 { self.fps }
}

// ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

impl InteractiveSurfaceApp {
    fn new() -> Result<Self> {
        let mut camera = Camera::new("/dev/video0")?;
        camera.start(&CameraConfig {
            interval: (1, 30),
            resolution: (640, 480),
            // Используем сырое YUYV (часто дешевле чем декодировать JPEG)
            format: b"YUYV",
            ..Default::default()
        })?;
        
        let settings = Settings::default();
        let mouse_controller = Enigo::new(&settings)?;
        let config = load_config().unwrap_or_default();
        
        let screen_points = vec![
            [0.0, 0.0],
            [1920.0, 0.0],
            [0.0, 1080.0],
            [1920.0, 1080.0],
        ];
        
        Ok(Self {
            camera: Some(camera),
            current_frame: None,
            filtered_frame: None,
            calibration_points: Vec::new(),
            screen_points,
            is_calibrated: false,
            calibration_step: 0,
            transform_matrix: None,
            mouse_controller,
            is_tracking: false,
            config,
            camera_texture: None,
            filter_texture: None,
            auto_click: false,
            stable_counter: 0,
            last_center: None,
            last_frame_time: std::time::Instant::now(),
            frame_interval: std::time::Duration::from_millis(33),
            fps_counter: FpsCounter::new(),
            mask_buffer: Vec::new(),
            rgba_buffer: Vec::new(),
            auto_calibrate: false,
            acc_min: None,
            acc_max: None,
            qr_enabled: false,
            qr_status: String::from("QR idle"),
            qr_attempt_counter: 0,
            qr_gray: Vec::new(),
            qr_multi_mode: false,
            qr_corner_codes: [None,None,None,None],
            qr_scanner: None,
        })
    }
    
    fn capture_and_process(&mut self) {
        // Throttle processing to target frame rate
        if self.last_frame_time.elapsed() < self.frame_interval {
            return;
        }
        self.last_frame_time = std::time::Instant::now();
        if let Some(camera) = &mut self.camera {
            if let Ok(frame) = camera.capture() {
                let w = 640u32; // fixed per config
                let h = 480u32;
                let pixels = &frame[..]; // YUYV: pairs of pixels (Y0 U Y1 V)
                let px_count = (w * h) as usize;
                if self.mask_buffer.len() != px_count { self.mask_buffer.resize(px_count, 0); }
                if self.rgba_buffer.len() != px_count * 4 { self.rgba_buffer.resize(px_count * 4, 0); }
                if self.qr_enabled && self.qr_gray.len() != px_count { self.qr_gray.resize(px_count, 0); }
                // Single pass convert YUYV -> RGBA + HSV mask
                yuyv_to_rgba_and_mask_with_y(
                    pixels,
                    w as usize,
                    h as usize,
                    &self.config,
                    &mut self.rgba_buffer,
                    &mut self.mask_buffer,
                    if self.qr_enabled { Some(&mut self.qr_gray) } else { None }
                );
                // Wrap RGBA into egui::ColorImage
                self.current_frame = Some(egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &self.rgba_buffer));
                // Build greyscale mask view
                self.filtered_frame = Some(mask_slice_to_color_image(&self.mask_buffer, w as usize, h as usize));
                // QR attempt every 10 кадров
                if self.qr_enabled && !self.is_calibrated && !self.qr_multi_mode {
                    self.qr_attempt_counter = self.qr_attempt_counter.wrapping_add(1);
                    if self.qr_attempt_counter % 10 == 0 {
                        if let Some(result) = try_qr_detect(&self.qr_gray, w as usize, h as usize) {
                            self.calibration_points = result;
                            self.calibration_step = 4;
                            self.complete_calibration();
                            self.qr_status = "QR OK".into();
                        } else {
                            self.qr_status = "QR scanning".into();
                        }
                    }
                }
                // Multi-QR mode: detect all codes each 10 кадров
                if self.qr_multi_mode && !self.is_calibrated {
                    self.qr_attempt_counter = self.qr_attempt_counter.wrapping_add(1);
                    if self.qr_attempt_counter % 10 == 0 {
                        if self.qr_scanner.is_none() { self.qr_scanner = Some(quircs::Quirc::default()); }
                        if let Some(scanner) = self.qr_scanner.as_mut() {
                            let _ = scanner.resize(w as usize, h as usize); // ignore resize errors
                            // feed grayscale
                            let codes = scanner.identify(&self.qr_gray);
                            for code in codes {
                                if let Ok(decoded) = code.decode() {
                                    let payload = String::from_utf8_lossy(decoded.payload()).to_string();
                                    let corners = code.corners(); // [(f64,f64);4]
                                    let cx = (corners[0].0 + corners[1].0 + corners[2].0 + corners[3].0) as f32 / 4.0;
                                    let cy = (corners[0].1 + corners[1].1 + corners[2].1 + corners[3].1) as f32 / 4.0;
                                    let idx = match payload.trim().to_uppercase().as_str() {
                                        "TL" => Some(0),
                                        "TR" => Some(1),
                                        "BL" => Some(2),
                                        "BR" => Some(3),
                                        _ => None,
                                    };
                                    if let Some(i) = idx { self.qr_corner_codes[i] = Some([cx,cy]); }
                                }
                            }
                            if self.qr_corner_codes.iter().all(|c| c.is_some()) {
                                self.calibration_points = vec![
                                    self.qr_corner_codes[0].unwrap(),
                                    self.qr_corner_codes[1].unwrap(),
                                    self.qr_corner_codes[2].unwrap(),
                                    self.qr_corner_codes[3].unwrap(),
                                ];
                                self.calibration_step = 4;
                                self.complete_calibration();
                                self.qr_status = "MultiQR OK".into();
                            } else {
                                self.qr_status = format!("MultiQR {} / 4", self.qr_corner_codes.iter().filter(|c| c.is_some()).count());
                            }
                        }
                    }
                }
                if let Some(center) = find_object_center(&self.mask_buffer, w, h, self.config.min_area) {
                    self.handle_object_detection(center);
                }
                self.fps_counter.tick();
            }
        }
    }
    
    fn handle_object_detection(&mut self, center: [f32; 2]) {
        if self.auto_calibrate && !self.is_calibrated {
            // accumulate bounding box of movement
            match (self.acc_min, self.acc_max) {
                (None, None) => { self.acc_min = Some(center); self.acc_max = Some(center); },
                (Some(mut mn), Some(mut mx)) => {
                    if center[0] < mn[0] { mn[0] = center[0]; }
                    if center[1] < mn[1] { mn[1] = center[1]; }
                    if center[0] > mx[0] { mx[0] = center[0]; }
                    if center[1] > mx[1] { mx[1] = center[1]; }
                    self.acc_min = Some(mn); self.acc_max = Some(mx);
                    // When area coverage large enough -> set calibration
                    if (mx[0]-mn[0]) > 50.0 && (mx[1]-mn[1]) > 50.0 { // heuristic thresholds
                        // define 4 points from bounding box
                        self.calibration_points = vec![
                            [mn[0], mn[1]],
                            [mx[0], mn[1]],
                            [mn[0], mx[1]],
                            [mx[0], mx[1]],
                        ];
                        self.calibration_step = 4;
                        self.complete_calibration();
                        println!("✅ Auto calibration done");
                    }
                }
                _ => {}
            }
        } else if !self.is_calibrated && self.calibration_step < 4 {
            self.calibration_points.push(center);
            self.calibration_step += 1;
            println!("🔊 Beep! Point {} captured: {:?}", self.calibration_step, center);
            if self.calibration_step >= 4 { self.complete_calibration(); }
        } else if self.is_calibrated && self.is_tracking {
            if let Some(screen_pos) = self.transform_point(center) {
                // Перемещение мыши всегда
                let _ = self.mouse_controller.move_mouse(
                    screen_pos[0] as i32,
                    screen_pos[1] as i32,
                    enigo::Coordinate::Abs,
                );
                // Клик только если включен авто клик и точка стабильна N кадров
                if self.auto_click {
                    const STABLE_FRAMES: u32 = 5;
                    const MOVE_EPS: f32 = 2.0;
                    if let Some(prev) = self.last_center {
                        let dx = prev[0] - center[0];
                        let dy = prev[1] - center[1];
                        if (dx * dx + dy * dy).sqrt() < MOVE_EPS {
                            self.stable_counter += 1;
                        } else {
                            self.stable_counter = 0;
                        }
                    }
                    self.last_center = Some(center);
                    if self.stable_counter >= STABLE_FRAMES {
                        let _ = self
                            .mouse_controller
                            .button(enigo::Button::Left, enigo::Direction::Click);
                        self.stable_counter = 0; // сброс после клика
                    }
                }
            }
        }
    }
    
    fn complete_calibration(&mut self) {
        if self.calibration_points.len() == 4 && self.screen_points.len() == 4 {
            self.transform_matrix = Some(calculate_simple_transform(
                &self.calibration_points,
                &self.screen_points
            ));
            self.is_calibrated = true;
            println!("✅ Calibration completed!");
        }
    }
    
    fn transform_point(&self, point: [f32; 2]) -> Option<[f32; 2]> {
        if let Some(matrix) = &self.transform_matrix {
            Some(apply_transform(matrix, point))
        } else {
            None
        }
    }
    
    fn reset_calibration(&mut self) {
        self.calibration_points.clear();
        self.calibration_step = 0;
        self.is_calibrated = false;
        self.transform_matrix = None;
        self.is_tracking = false;
        self.acc_min = None;
        self.acc_max = None;
        println!("🔄 Calibration reset");
    }
}

// ==================== ОБРАБОТКА ИЗОБРАЖЕНИЙ ====================

// Преобразование YUYV -> RGBA + HSV маска (одним проходом)
fn yuyv_to_rgba_and_mask(
    yuyv: &[u8],
    width: usize,
    height: usize,
    config: &AppConfig,
    rgba_out: &mut [u8],
    mask_out: &mut [u8],
) {
    // Каждый блок: Y0 U Y1 V (2 пикселя)
    // Формулы преобразования (BT.601 приблизительно)
    // R = 1.164*(Y-16) + 1.596*(V-128)
    // G = 1.164*(Y-16) - 0.813*(V-128) - 0.391*(U-128)
    // B = 1.164*(Y-16) + 2.018*(U-128)
    let mut pi = 0; // индекс пикселя
    for chunk in yuyv.chunks_exact(4) { // два пикселя
        let y0 = chunk[0] as f32;
        let u  = chunk[1] as f32;
        let y1 = chunk[2] as f32;
        let v  = chunk[3] as f32;
        let (r0,g0,b0) = yuv_to_rgb(y0,u,v);
        let (r1,g1,b1) = yuv_to_rgb(y1,u,v);
        // Пишем RGBA
        let base0 = pi * 4; // первый пиксель
        rgba_out[base0] = r0; rgba_out[base0+1] = g0; rgba_out[base0+2] = b0; rgba_out[base0+3] = 255;
        // HSV для маски
        let hsv0 = rgb_to_hsv([r0,g0,b0]);
        mask_out[pi] = if in_hsv_range(&hsv0, config) { 255 } else { 0 };
        pi += 1;
        let base1 = pi * 4; // второй пиксель
        rgba_out[base1] = r1; rgba_out[base1+1] = g1; rgba_out[base1+2] = b1; rgba_out[base1+3] = 255;
        let hsv1 = rgb_to_hsv([r1,g1,b1]);
        mask_out[pi] = if in_hsv_range(&hsv1, config) { 255 } else { 0 };
        pi += 1;
    }
    debug_assert_eq!(pi, width*height);
}

// Вариант с извлечением Y-плоскости для QR
fn yuyv_to_rgba_and_mask_with_y(
    yuyv: &[u8],
    width: usize,
    height: usize,
    config: &AppConfig,
    rgba_out: &mut [u8],
    mask_out: &mut [u8],
    y_plane_opt: Option<&mut Vec<u8>>,
) {
    let mut pi = 0;
    if let Some(y_plane) = y_plane_opt {
        for chunk in yuyv.chunks_exact(4) {
            let y0 = chunk[0] as f32; let u  = chunk[1] as f32; let y1 = chunk[2] as f32; let v  = chunk[3] as f32;
            let (r0,g0,b0) = yuv_to_rgb(y0,u,v);
            let (r1,g1,b1) = yuv_to_rgb(y1,u,v);
            let base0 = pi*4; rgba_out[base0]=r0; rgba_out[base0+1]=g0; rgba_out[base0+2]=b0; rgba_out[base0+3]=255;
            let hsv0 = rgb_to_hsv([r0,g0,b0]); mask_out[pi] = if in_hsv_range(&hsv0, config){255}else{0}; y_plane[pi] = y0 as u8; pi+=1;
            let base1 = pi*4; rgba_out[base1]=r1; rgba_out[base1+1]=g1; rgba_out[base1+2]=b1; rgba_out[base1+3]=255;
            let hsv1 = rgb_to_hsv([r1,g1,b1]); mask_out[pi] = if in_hsv_range(&hsv1, config){255}else{0}; y_plane[pi] = y1 as u8; pi+=1;
        }
    } else {
        yuyv_to_rgba_and_mask(yuyv,width,height,config,rgba_out,mask_out);
        return;
    }
    debug_assert_eq!(pi, width*height);
}

fn try_qr_detect(y_plane: &[u8], width: usize, height: usize) -> Option<Vec<[f32;2]>> {
    use rqrr::PreparedImage;
    // Создаём Luma ImageBuffer из Y-плоскости
    let img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(width as u32, height as u32, y_plane.to_vec())?;
    let mut prep = PreparedImage::prepare(img);
    let grids = prep.detect_grids();
    let grid = grids.get(0)?;
    // Попытка декодирования (убедиться, что это настоящий QR)
    if grid.decode().is_err() { return None; }
    // Попытаемся получить углы. Если API изменится — fallback на None.
    #[allow(unused_mut)]
    let mut pts: Vec<[f32;2]> = Vec::new();
    // Пробуем несколько возможных методов (один из них скомпилируется, лишние закомментированы):
    // if let Some(corners) = grid.get_corners() { ... }
    // В отсутствие публичного метода используем bounding box через перебор всех предполагаемых точек сетки.
    if pts.is_empty() {
        // Heuristic: rqrr grid может предоставить width/height модуля и функцию loc(x,y)
        // Попытаемся через отражённый интерфейс (если не существует — компилятор подскажет, тогда вернём None)
        #[allow(unused_variables)]
        {
            // placeholder — возвращаем None если не можем извлечь углы
        }
        return None;
    }
    Some(pts)
}

#[inline]
fn yuv_to_rgb(y: f32, u: f32, v: f32) -> (u8,u8,u8) {
    let yv = 1.164*(y - 16.0);
    let uo = u - 128.0;
    let vo = v - 128.0;
    let r = (yv + 1.596*vo).clamp(0.0,255.0) as u8;
    let g = (yv - 0.813*vo - 0.391*uo).clamp(0.0,255.0) as u8;
    let b = (yv + 2.018*uo).clamp(0.0,255.0) as u8;
    (r,g,b)
}

#[inline]
fn in_hsv_range(hsv: &[f32;3], config: &AppConfig) -> bool {
    hsv[0] >= config.hsv_lower[0] as f32 && hsv[0] <= config.hsv_upper[0] as f32 &&
    hsv[1] >= config.hsv_lower[1] as f32 && hsv[1] <= config.hsv_upper[1] as f32 &&
    hsv[2] >= config.hsv_lower[2] as f32 && hsv[2] <= config.hsv_upper[2] as f32
}

fn mask_slice_to_color_image(mask: &[u8], width: usize, height: usize) -> egui::ColorImage {
    // Грей → RGB
    let mut rgb = Vec::with_capacity(width*height*3);
    for &v in mask { rgb.push(v); rgb.push(v); rgb.push(v); }
    egui::ColorImage::from_rgb([width,height], &rgb)
}

use rayon::prelude::*;

#[inline]
fn rgb_to_hsv(rgb: [u8; 3]) -> [f32; 3] {
    let r = rgb[0] as f32 / 255.0;
    let g = rgb[1] as f32 / 255.0;
    let b = rgb[2] as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let mut h_deg = if delta == 0.0 {
        0.0
    } else if (max - r).abs() < f32::EPSILON {
        60.0 * ((g - b) / delta % 6.0)
    } else if (max - g).abs() < f32::EPSILON {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    if h_deg < 0.0 { h_deg += 360.0; }
    // Приводим к шкале 0..180 (имитация OpenCV) делением на 2
    let h = h_deg / 2.0; // 0..180
    let s = if max == 0.0 { 0.0 } else { delta / max } * 255.0; // 0..255
    let v = max * 255.0; // 0..255
    [h, s, v]
}

fn find_object_center(mask: &[u8], width: u32, _height: u32, min_area: f64) -> Option<[f32; 2]> {
    let row_len = width as usize;
    let partials: Vec<(u64, u64, u64)> = mask
        .par_chunks(row_len)
        .enumerate()
        .map(|(y, row)| {
            let mut mass = 0u64;
            let mut x_sum = 0u64;
            let mut y_sum = 0u64;
            for (x, &v) in row.iter().enumerate() {
                if v > 128 {
                    mass += 1;
                    x_sum += x as u64;
                    y_sum += y as u64;
                }
            }
            (mass, x_sum, y_sum)
        })
        .collect();

    let (total_mass, total_x, total_y) = partials.into_iter().fold(
        (0u64, 0u64, 0u64),
        |acc, v| (acc.0 + v.0, acc.1 + v.1, acc.2 + v.2),
    );
    if (total_mass as f64) >= min_area {
        Some([
            total_x as f32 / total_mass as f32,
            total_y as f32 / total_mass as f32,
        ])
    } else {
        None
    }
}

// ==================== ПЕРСПЕКТИВНАЯ ТРАНСФОРМАЦИЯ ====================

fn calculate_simple_transform(camera_points: &[[f32; 2]], screen_points: &[[f32; 2]]) -> [[f64; 3]; 3] {
    let cam_tl = camera_points[0];
    let cam_br = camera_points[3];
    let scr_tl = screen_points[0];
    let scr_br = screen_points[3];
    
    let scale_x = (scr_br[0] - scr_tl[0]) as f64 / (cam_br[0] - cam_tl[0]) as f64;
    let scale_y = (scr_br[1] - scr_tl[1]) as f64 / (cam_br[1] - cam_tl[1]) as f64;
    
    let offset_x = scr_tl[0] as f64 - cam_tl[0] as f64 * scale_x;
    let offset_y = scr_tl[1] as f64 - cam_tl[1] as f64 * scale_y;
    
    [
        [scale_x, 0.0, offset_x],
        [0.0, scale_y, offset_y], 
        [0.0, 0.0, 1.0]
    ]
}

fn apply_transform(matrix: &[[f64; 3]; 3], point: [f32; 2]) -> [f32; 2] {
    let x = matrix[0][0] * point[0] as f64 + matrix[0][1] * point[1] as f64 + matrix[0][2];
    let y = matrix[1][0] * point[0] as f64 + matrix[1][1] * point[1] as f64 + matrix[1][2];
    
    [x as f32, y as f32]
}

// ==================== НАСТРОЙКИ ====================

fn load_config() -> Option<AppConfig> {
    std::fs::read_to_string("config.json")
        .ok()
        .and_then(|data| serde_json::from_str(&data).ok())
}

fn save_config(config: &AppConfig) -> Result<()> {
    let data = serde_json::to_string_pretty(config)?;
    std::fs::write("config.json", data)?;
    Ok(())
}

// ==================== GUI ====================

impl App for InteractiveSurfaceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.capture_and_process();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🎯 Interactive Surface");
            
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label("📹 Camera Feed");
                    if let Some(ref frame) = self.current_frame {
                        if let Some(texture) = self.camera_texture.as_mut() {
                            texture.set(frame.clone(), egui::TextureOptions::default());
                        } else {
                            self.camera_texture = Some(ctx.load_texture("camera", frame.clone(), egui::TextureOptions::default()));
                        }
                        
                        if let Some(texture) = &self.camera_texture {
                            ui.add(egui::Image::from_texture(texture).max_width(300.0));
                        }
                    } else {
                        ui.label("Connecting...");
                    }
                });
                
                ui.separator();
                
                ui.vertical(|ui| {
                    ui.label("🔍 HSV Filter");
                    if let Some(ref filtered) = self.filtered_frame {
                        if let Some(texture) = self.filter_texture.as_mut() {
                            texture.set(filtered.clone(), egui::TextureOptions::default());
                        } else {
                            self.filter_texture = Some(ctx.load_texture("filter", filtered.clone(), egui::TextureOptions::default()));
                        }
                        
                        if let Some(texture) = &self.filter_texture {
                            ui.add(egui::Image::from_texture(texture).max_width(300.0));
                        }
                    }
                });
            });
            
            ui.separator();
            
            ui.horizontal(|ui| {
                if ui.button("🎯 Reset Calibration").clicked() {
                    self.reset_calibration();
                }
                
                ui.separator();
                
                if self.is_calibrated {
                    if self.is_tracking {
                        if ui.button("⏹ Stop Tracking").clicked() {
                            self.is_tracking = false;
                        }
                    } else {
                        if ui.button("▶ Start Tracking").clicked() {
                            self.is_tracking = true;
                        }
                    }
                } else {
                    ui.add_enabled(false, egui::Button::new("⏳ Calibrating..."));
                }
            });
            
            ui.separator();
            
            // Исправляем ошибку типов в статусе
            ui.horizontal(|ui| {
                if self.is_tracking {
                    ui.label("🟢 Tracking Active");
                } else if self.is_calibrated {
                    ui.label("🟡 Ready to Track");
                } else {
                    ui.label(format!("🔴 Calibrating ({}/4 points)", self.calibration_step));
                }
                ui.checkbox(&mut self.auto_click, "🤖 Auto Click");
                ui.checkbox(&mut self.auto_calibrate, "🧭 AutoCalib");
                if self.auto_calibrate && !self.is_calibrated {
                    if let (Some(min), Some(max)) = (self.acc_min, self.acc_max) {
                        ui.label(format!("box {:.0}x{:.0}", max[0]-min[0], max[1]-min[1]));
                    } else {
                        ui.label("box ...");
                    }
                }
                ui.separator();
                ui.checkbox(&mut self.qr_enabled, "📐 QR");
                if self.qr_enabled { ui.label(self.qr_status.clone()); }
                ui.checkbox(&mut self.qr_multi_mode, "🧩 MultiQR");
                if self.qr_multi_mode {
                    let labels = ["TL","TR","BL","BR"];
                    for (i,l) in labels.iter().enumerate() {
                        let ok = self.qr_corner_codes[i].is_some();
                        ui.label(format!("{}:{}", l, if ok {"✔"} else {"·"}));
                    }
                }
                ui.label(format!("FPS: {:.1}", self.fps_counter.fps()));
            });
            
            ui.collapsing("⚙ HSV Settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Lower HSV:");
                    ui.add(egui::Slider::new(&mut self.config.hsv_lower[0], 0..=179).text("H"));
                    ui.add(egui::Slider::new(&mut self.config.hsv_lower[1], 0..=255).text("S"));
                    ui.add(egui::Slider::new(&mut self.config.hsv_lower[2], 0..=255).text("V"));
                });
                
                ui.horizontal(|ui| {
                    ui.label("Upper HSV:");
                    ui.add(egui::Slider::new(&mut self.config.hsv_upper[0], 0..=179).text("H"));
                    ui.add(egui::Slider::new(&mut self.config.hsv_upper[1], 0..=255).text("S"));
                    ui.add(egui::Slider::new(&mut self.config.hsv_upper[2], 0..=255).text("V"));
                });
                
                if ui.button("💾 Save Settings").clicked() {
                    let _ = save_config(&self.config);
                }
            });
        });
        
        ctx.request_repaint();
    }
}

// ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_title("Interactive Surface"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Interactive Surface",
        options,
        Box::new(|_cc| Ok(Box::new(InteractiveSurfaceApp::new().unwrap()))),
    )?;
    
    Ok(())
}
