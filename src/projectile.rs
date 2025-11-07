use glam::Vec3;

#[derive(Default, Debug)]
pub struct Projectile {
    pub active: bool,
    pub position: Vec3,
    pub velocity: Vec3,
    pub lifetime: f32,
}