use glam::{Mat4, Vec3};

const PITCH_MIN: f32 = -std::f32::consts::FRAC_PI_2 + 0.01;
const PITCH_MAX: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

#[derive(Debug)]
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub up: Vec3,
}

impl Camera {
    pub fn new(position: Vec3) -> Self {
        Self {
            position,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            up: Vec3::Y,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        ).normalize()
    }

    pub fn view_matrix(&self) -> Mat4 {
        let target = self.position + self.forward();
        Mat4::look_at_rh(self.position, target, self.up)
    }

    pub fn clamp_pitch(&mut self) {
        self.pitch = self.pitch.clamp(PITCH_MIN, PITCH_MAX);
    }
}