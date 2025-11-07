use glam::Mat4;

#[derive(Default)]
pub struct TransformStack {
    stack: Vec<Mat4>,
}

impl TransformStack {
    pub fn new() -> Self {
        Self {
            stack: vec![Mat4::IDENTITY],
        }
    }

    pub fn push(&mut self, transform: Mat4) {
        let current = *self.stack.last().unwrap();
        self.stack.push(current * transform);
    }

    pub fn pop(&mut self) {
        if self.stack.len() > 1 {
            self.stack.pop();
        }
    }

    pub fn current(&self) -> Mat4 {
        *self.stack.last().unwrap()
    }
}