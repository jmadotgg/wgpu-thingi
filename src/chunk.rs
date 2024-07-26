use std::collections::HashMap;

use crate::{Vertex, INDICES};

const CHUNK_SIZE: usize = 64;
const HALF_CHUNK_SIZE: i32 = (CHUNK_SIZE / 2) as i32;
const CHUNK_DISPLACEMENT: f32 = (CHUNK_SIZE / 2) as f32;

#[derive(Debug, Copy, Clone, PartialEq)]
enum Block {
    GRASS = 1,
    AIR = 0,
}

#[derive(Debug, PartialEq, Hash, Clone, Copy)]
pub struct Position {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl std::cmp::Eq for Position {}

impl Position {
    pub fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    pub fn is_in_chunk(&self, chunk_pos: &Position) -> bool {
        if self.x > (chunk_pos.x + HALF_CHUNK_SIZE) || self.x < (chunk_pos.x - HALF_CHUNK_SIZE) {
            return false;
        }
        if self.y > (chunk_pos.y + HALF_CHUNK_SIZE) || self.y < (chunk_pos.y - HALF_CHUNK_SIZE) {
            return false;
        }
        if self.z > (chunk_pos.z + HALF_CHUNK_SIZE) || self.z < (chunk_pos.z - HALF_CHUNK_SIZE) {
            return false;
        }
        return true;
    }
}

impl From<cgmath::Point3<f32>> for Position {
    fn from(value: cgmath::Point3<f32>) -> Self {
        Self::new(value.x as i32, value.y as i32, value.z as i32)
    }
}

#[derive(Debug)]
pub struct Chunk {
    pos: Position,
    chunk_data: [Block; CHUNK_SIZE.pow(3)],
}

pub fn chunk_position_from_player_position(player_position: &Position) -> Position {
    let x = (player_position.x / CHUNK_SIZE as i32) * CHUNK_SIZE as i32
        + if player_position.x < 0 { -1 } else { 1 } * HALF_CHUNK_SIZE;

    let y = (player_position.y / CHUNK_SIZE as i32) * CHUNK_SIZE as i32
        + if player_position.y < 0 { -1 } else { 1 } * HALF_CHUNK_SIZE;

    let z = (player_position.z / CHUNK_SIZE as i32) * CHUNK_SIZE as i32
        + if player_position.z < 0 { -1 } else { 1 } * HALF_CHUNK_SIZE;

    return Position::new(x, y, z);
}

impl Chunk {
    pub fn new(pos: &Position) -> Self {
        let mut chunk_data = [Block::GRASS; CHUNK_SIZE.pow(3)];
        for i in 0..chunk_data.len() {
            //chunk_data[i] = Block::GRASS;
            chunk_data[i] = match rand::random() {
                0.0..0.5 => Block::AIR,
                0.5..1.0 => Block::GRASS,
                _ => todo!(),
            }
        }
        Self {
            pos: pos.clone(),
            chunk_data,
        }
    }

    pub fn generate_chunk_data(&mut self) {
        todo!()
    }
}

pub struct Mesh {
    pub indices: Vec<u32>,
    pub vertices: Vec<Vertex>,
}

impl Mesh {
    fn fancy_algorithm_to_calculate_mesh_from_chunk() {
        todo!()
    }
}

impl From<&Chunk> for Mesh {
    fn from(value: &Chunk) -> Self {
        let Chunk { pos, chunk_data } = value;

        let color: [[f32; 3]; 8] = [
            [0.5, 0.0, 0.2],
            [0.0, 0.5, 0.2],
            [0.0, 0.0, 0.2],
            [0.5, 0.0, 0.2],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
        ];

        let mut vertices = Vec::new();

        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                // order of x and y very important for vertex positions
                for z in 0..CHUNK_SIZE {
                    let block = chunk_data[y * z + x];

                    if block == Block::GRASS {
                        let (x, y, z) = (
                            pos.x as f32 - CHUNK_DISPLACEMENT + x as f32,
                            pos.y as f32 - CHUNK_DISPLACEMENT + y as f32,
                            pos.z as f32 - CHUNK_DISPLACEMENT + z as f32,
                        );

                        // move to chunk generation
                        if y > 20.0 {
                            continue;
                        }

                        #[rustfmt::skip]
                        vertices.push(vec![
                            Vertex { position: [x + -0.5, y + 0.5,  z + 0.0], color: color[0] }, // A
                            Vertex { position: [x + -0.5, y + -0.5, z + 0.0], color: color[1] }, // B
                            Vertex { position: [x + 0.5,  y + 0.5,  z + 0.0], color: color[2] }, // C
                            Vertex { position: [x + 0.5,  y + -0.5, z + 0.0], color: color[3] }, // D
                            Vertex { position: [x + -0.5, y + 0.5,  z + -1.0], color: color[4] }, // A
                            Vertex { position: [x + -0.5, y + -0.5, z + -1.0], color: color[5] }, // B
                            Vertex { position: [x + 0.5,  y + 0.5,  z + -1.0], color: color[6] }, // C
                            Vertex { position: [x + 0.5,  y + -0.5, z + -1.0], color: color[7] }, // D
                        ]);
                    }
                }
            }
        }

        let indices = (0..vertices.len())
            .flat_map(|i| {
                INDICES
                    .to_vec()
                    .iter()
                    .map(|e| i as u32 * 8 as u32 + *e as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let vertices = vertices.into_iter().flatten().collect::<Vec<_>>();

        Self { vertices, indices }
    }
}

pub struct ChunkWatcher {
    chunks: HashMap<Position, Chunk>,
    meshes: HashMap<Position, Mesh>,
}

impl ChunkWatcher {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            meshes: HashMap::new(),
        }
    }

    pub fn update_buffers(
        &mut self,
        queue: &mut wgpu::Queue,
        vertex_buffer: &wgpu::Buffer,
        index_buffer: &wgpu::Buffer,
        num_indices: &mut u32,
        player_position: &Position,
        current_chunk: &mut Position,
    ) {
        if player_position.is_in_chunk(&current_chunk) {
            return;
        }

        *current_chunk = chunk_position_from_player_position(player_position);

        if let Some(mesh) = self.meshes.get(current_chunk) {
            queue.write_buffer(vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));
            queue.write_buffer(index_buffer, 0, bytemuck::cast_slice(&mesh.indices));
            *num_indices = mesh.indices.len() as u32;
            return;
        }

        let chunk = Chunk::new(current_chunk);
        let mesh = Mesh::from(&chunk);
        *num_indices = mesh.indices.len() as u32;

        queue.write_buffer(vertex_buffer, 0, bytemuck::cast_slice(&mesh.vertices));
        queue.write_buffer(index_buffer, 0, bytemuck::cast_slice(&mesh.indices));

        self.chunks.insert(current_chunk.clone(), chunk);
        self.meshes.insert(current_chunk.clone(), mesh);
    }

    fn get_mesh(&self, pos: &Position) -> Mesh {
        //if let Some(mesh) = self.meshes.get(&pos) {
        //    return *mesh;
        //}

        let chunk = Chunk::new(pos);
        let mesh = Mesh::from(&chunk);

        //self.chunks.insert(pos.clone(), chunk);
        //self.meshes.insert(pos.clone(), mesh).unwrap();

        mesh
    }

    pub fn get_required_chunk_mesh_data(&mut self, player_position: &Position) -> Vec<Mesh> {
        let positions = get_required_chunk_positions(player_position);

        positions
            .iter()
            .map(|pos| self.get_mesh(pos))
            .collect::<Vec<_>>()
    }
}

pub fn get_required_chunk_positions(player_position: &Position) -> Vec<Position> {
    let mut positions = Vec::new();
    for y in -VIEW_DISTANCE..VIEW_DISTANCE {
        for z in -VIEW_DISTANCE..VIEW_DISTANCE {
            for x in -VIEW_DISTANCE..VIEW_DISTANCE {
                positions.push(Position::new(
                    player_position.x + x * CHUNK_SIZE as i32,
                    player_position.y + y * CHUNK_SIZE as i32,
                    player_position.z + z * CHUNK_SIZE as i32,
                ));
            }
        }
    }
    positions
}

pub const VIEW_DISTANCE: i32 = 2;
pub const CHUNK_COUNT: i32 = (VIEW_DISTANCE * 2).pow(3);
pub const CHUNK_MAX_VERTEX_COUNT: usize = CHUNK_SIZE.pow(3) * 8;
pub const CHUNK_MAX_INDEX_COUNT: usize = 37748736; //CHUNK_SIZE.pow(3) * 36;
