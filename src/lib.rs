use std::{borrow::Cow, collections::HashMap};

use inline_spirv::include_spirv;
use screen_13::graph::Bind;

pub use {
    archery::{ArcK, SharedPointer, SharedPointerKind},
    bytemuck::cast_slice,
    screen_13::prelude_arc::*,
};

pub struct Egui {
    pipeline: SharedPointer<GraphicPipeline, ArcK>,
    pool: HashPool,

    shapes: Vec<egui::epaint::ClippedShape>,
    tex_delta: egui::TexturesDelta,
    egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    textures: HashMap<egui::TextureId, ImageLeaseBinding<ArcK>>,
}

impl Egui {
    pub fn new(device: &SharedPointer<Device, ArcK>, window: &Window) -> Self {
        let pool = HashPool::new(device);
        let pipeline = SharedPointer::new(
            GraphicPipeline::create(
                device,
                GraphicPipelineInfo::new()
                    .blend(BlendMode::PreMultipliedAlpha)
                    .cull_mode(vk::CullModeFlags::NONE),
                [
                    Shader::new_vertex(
                        include_spirv!("shaders/vert.glsl", vert, vulkan1_2).as_slice(),
                    ),
                    Shader::new_fragment(
                        include_spirv!("shaders/frag.glsl", frag, vulkan1_2).as_slice(),
                    ),
                ],
            )
            .unwrap(),
        );

        let egui_winit = egui_winit::State::new(10000, window);

        Self {
            pool,
            pipeline,
            shapes: Vec::default(),
            tex_delta: egui::TexturesDelta::default(),
            egui_ctx: egui::Context::default(),
            egui_winit,
            textures: HashMap::default(),
        }
    }

    pub fn run(
        &mut self,
        window: &Window,
        dst: impl Into<AnyImageNode>,
        render_graph: &mut RenderGraph,
        ui_fn: impl FnMut(&egui::Context),
    ) {
        let dst = dst.into();

        let raw_input = self.egui_winit.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, ui_fn);

        self.egui_winit
            .handle_platform_output(window, &self.egui_ctx, full_output.platform_output);

        self.shapes = full_output.shapes;
        self.tex_delta.append(full_output.textures_delta);

        if full_output.needs_repaint {
            let shapes = std::mem::take(&mut self.shapes);
            let tex_delta = std::mem::take(&mut self.tex_delta);
            let clipped_primitives = self.egui_ctx.tessellate(shapes);

            let mut bound_tex: HashMap<egui::TextureId, ImageLeaseNode<ArcK>> = HashMap::default();

            for (id, delta) in tex_delta.set.iter() {
                bound_tex.insert(*id, self.set_texture(id, delta, render_graph));
            }

            self.paint_primitives(
                dst,
                render_graph,
                self.egui_ctx.pixels_per_point(),
                &bound_tex,
                &clipped_primitives,
            );

            for id in tex_delta.free {
                self.textures.remove(&id);
            }

            for (id, tex) in bound_tex.iter() {
                self.textures.insert(*id, render_graph.unbind_node(*tex));
            }
        }
    }

    fn paint_primitives(
        &mut self,
        target: AnyImageNode,
        render_graph: &mut RenderGraph,
        ppp: f32,
        bound_tex: &HashMap<egui::TextureId, ImageLeaseNode<ArcK>>,
        clipped_primitives: &[egui::ClippedPrimitive],
    ) {
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_primitives
        {
            match primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    self.paint_mesh(target, render_graph, ppp, *clip_rect, bound_tex, mesh);
                }
                egui::epaint::Primitive::Callback(callback) => {
                    panic!("Primitive callback not yet supported.");
                }
            }
        }
    }

    fn paint_mesh(
        &mut self,
        target: AnyImageNode,
        render_graph: &mut RenderGraph,
        ppp: f32,
        clip_rect: egui::epaint::Rect,
        bound_tex: &HashMap<egui::TextureId, ImageLeaseNode<ArcK>>,
        mesh: &egui::epaint::Mesh,
    ) {
        if let Some(texture) = bound_tex.get(&mesh.texture_id) {
            let idx_buf = {
                let mut buf = self
                    .pool
                    .lease(BufferInfo::new_mappable(
                        (mesh.indices.len() * 4) as u64,
                        vk::BufferUsageFlags::INDEX_BUFFER,
                    ))
                    .unwrap();
                Buffer::copy_from_slice(buf.get_mut().unwrap(), 0, cast_slice(&mesh.indices));
                buf
            };
            let idx_buf = render_graph.bind_node(idx_buf);

            let vert_buf = {
                let mut buf = self
                    .pool
                    .lease(BufferInfo::new_mappable(
                        (mesh.vertices.len() * std::mem::size_of::<egui::epaint::Vertex>()) as u64,
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                    ))
                    .unwrap();
                Buffer::copy_from_slice(buf.get_mut().unwrap(), 0, cast_slice(&mesh.vertices));
                buf
            };
            let vert_buf = render_graph.bind_node(vert_buf);

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct PushConstants {
                screen_size: [f32; 2],
            }

            let target_info = render_graph.node_info(target);

            let push_constants = PushConstants {
                screen_size: [
                    target_info.width as f32 / ppp,
                    target_info.height as f32 / ppp,
                ],
            };

            let num_indices = mesh.indices.len() as u32;
            trace!("target_idx: {:#?}", target);
            let pass = render_graph
                .begin_pass("Egui pass")
                .bind_pipeline(&self.pipeline)
                .read_descriptor((0, 0), *texture)
                .access_node(idx_buf, AccessType::IndexBuffer)
                .access_node(vert_buf, AccessType::VertexBuffer)
                .clear_color(0)
                .store_color(0, target)
                .record_subpass(move |subpass| {
                    subpass.bind_index_buffer(idx_buf, vk::IndexType::UINT32);
                    subpass.bind_vertex_buffer(vert_buf);
                    subpass.push_constants(cast_slice(&[push_constants]));
                    subpass.set_scissor(
                        clip_rect.min.x as i32,
                        clip_rect.min.y as i32,
                        (clip_rect.max.x - clip_rect.min.x) as u32,
                        (clip_rect.max.y - clip_rect.min.y) as u32,
                    );
                    subpass.draw_indexed(num_indices, 1, 0, 0, 0);
                });


            render_graph.unbind_node(idx_buf);
            render_graph.unbind_node(vert_buf);
        }
    }

    fn set_texture(
        &mut self,
        tex_id: &egui::TextureId,
        delta: &egui::epaint::ImageDelta,
        render_graph: &mut RenderGraph,
    ) -> ImageLeaseNode<ArcK> {
        let pixels = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(image.width() * image.height(), image.pixels.len());
                Cow::Borrowed(&image.pixels)
            }
            egui::ImageData::Font(image) => {
                let gamma = 1.0;
                Cow::Owned(image.srgba_pixels(gamma).collect::<Vec<_>>())
            }
        };

        let tmp_buf = {
            let mut buf = self
                .pool
                .lease(BufferInfo::new_mappable(
                    (pixels.len() * 4) as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                ))
                .unwrap();
            Buffer::copy_from_slice(buf.get_mut().unwrap(), 0, cast_slice(&pixels));
            render_graph.bind_node(buf)
        };

        if let Some(pos) = delta.pos {
            let image = self
                .textures
                .remove(&tex_id)
                .expect("tried to update non existing texture.")
                .bind(render_graph);
            render_graph.copy_buffer_to_image_region(
                tmp_buf,
                image,
                &vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_offset: vk::Offset3D {
                        x: pos[0] as i32,
                        y: pos[0] as i32,
                        z: 0,
                    },
                    image_extent: vk::Extent3D {
                        width: delta.image.width() as u32,
                        height: delta.image.height() as u32,
                        depth: 1,
                    },
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                },
            );
            image
            //*self.textures.get_mut(&tex_id).unwrap() = Some(render_graph.unbind_node(image));
        } else {
            let image = ImageLeaseBinding({
                self.pool
                    .lease(ImageInfo::new_2d(
                        vk::Format::R8G8B8A8_UNORM,
                        delta.image.width() as u32,
                        delta.image.height() as u32,
                        vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::TRANSFER_DST,
                    ))
                    .unwrap()
            });
            let image = image.bind(render_graph);

            render_graph.copy_buffer_to_image(tmp_buf, image);

            image
        }
    }
}
