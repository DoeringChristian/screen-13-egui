use std::{borrow::Cow, collections::HashMap};

use screen_13::graph::Bind;

pub use {
    archery::{ArcK, SharedPointer, SharedPointerKind},
    bytemuck::cast_slice,
    screen_13::prelude_arc::*,
};

pub struct Egui {
    pub egui_ctx: egui::Context,
    egui_winit: egui_winit::State,
    textures: HashMap<egui::TextureId, ImageLeaseBinding<ArcK>>,
    cache: HashPool,
    ppl: SharedPointer<GraphicPipeline, ArcK>,
}

impl Egui {
    pub fn new(device: &SharedPointer<Device, ArcK>, window: &Window) -> Self {
        let ppl = SharedPointer::new(
            GraphicPipeline::create(
                device,
                GraphicPipelineInfo::new()
                    .blend(BlendMode {
                        blend_enable: vk::TRUE,
                        src_color_blend_factor: vk::BlendFactor::ONE,
                        dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        color_blend_op: vk::BlendOp::ADD,
                        src_alpha_blend_factor: vk::BlendFactor::ONE,
                        dst_alpha_blend_factor: vk::BlendFactor::ONE,
                        alpha_blend_op: vk::BlendOp::ADD,
                        // Have to construct it from raw bits since | operator is not const.
                        color_write_mask: vk::ColorComponentFlags::from_raw(0b1111),
                    })
                    .cull_mode(vk::CullModeFlags::NONE),
                [
                    Shader::new_vertex(
                        inline_spirv::include_spirv!("shaders/vert.glsl", vert, vulkan1_2)
                            .as_slice(),
                    ),
                    Shader::new_fragment(
                        inline_spirv::include_spirv!("shaders/frag.glsl", frag, vulkan1_2)
                            .as_slice(),
                    ),
                ],
            )
            .unwrap(),
        );
        Self {
            ppl,
            egui_ctx: egui::Context::default(),
            egui_winit: egui_winit::State::new(10000, window),
            textures: HashMap::default(),
            cache: HashPool::new(device),
        }
    }

    fn bind_and_update_textures(
        &mut self,
        deltas: &egui::TexturesDelta,
        render_graph: &mut RenderGraph,
    ) -> HashMap<egui::TextureId, ImageLeaseNode<ArcK>> {
        let mut bound_tex = deltas
            .set
            .iter()
            .map(|(id, delta)| {
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
                        .cache
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
                        .remove(&id)
                        .expect("Tried updating undefined texture.")
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
                                y: pos[1] as i32,
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
                    (*id, image)
                } else {
                    let image = self
                        .cache
                        .lease(ImageInfo::new_2d(
                            vk::Format::R8G8B8A8_UNORM,
                            delta.image.width() as u32,
                            delta.image.height() as u32,
                            vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::TRANSFER_DST,
                        ))
                        .unwrap()
                        .bind(render_graph);

                    render_graph.copy_buffer_to_image(tmp_buf, image);
                    render_graph.unbind_node(tmp_buf);
                    (*id, image)
                }
            })
            .collect::<HashMap<_, _>>();

        // Bind the rest of the textures.
        for (id, image) in self.textures.drain() {
            bound_tex.insert(id, render_graph.bind_node(image));
        }
        bound_tex
    }

    fn unbind_and_free(
        &mut self,
        bound_tex: HashMap<egui::TextureId, ImageLeaseNode<ArcK>>,
        render_graph: &mut RenderGraph,
        deltas: &egui::TexturesDelta,
    ) {
        // Unbind textures
        for (id, tex) in bound_tex.iter() {
            self.textures.insert(*id, render_graph.unbind_node(*tex));
        }

        // Free textures.
        for id in deltas.free.iter() {
            self.textures.remove(&id);
        }
    }

    fn draw_primitive(
        &mut self,
        shapes: Vec<egui::epaint::ClippedShape>,
        bound_tex: &HashMap<egui::TextureId, ImageLeaseNode<ArcK>>,
        render_graph: &mut RenderGraph,
        target: impl Into<AnyImageNode>,
    ) {
        let target = target.into();
        let target_info = render_graph.node_info(target);
        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in self.egui_ctx.tessellate(shapes)
        {
            match primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    // Skip when there are no vertices to paint since we cannot allocate a buffer
                    // of length 0
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    let texture = bound_tex.get(&mesh.texture_id).unwrap();

                    let idx_buf = {
                        let mut buf = self
                            .cache
                            .lease(BufferInfo::new_mappable(
                                (mesh.indices.len() * 4) as u64,
                                vk::BufferUsageFlags::INDEX_BUFFER,
                            ))
                            .unwrap();
                        Buffer::copy_from_slice(
                            buf.get_mut().unwrap(),
                            0,
                            cast_slice(&mesh.indices),
                        );
                        buf
                    };
                    let idx_buf = render_graph.bind_node(idx_buf);

                    let vert_buf = {
                        let mut buf = self
                            .cache
                            .lease(BufferInfo::new_mappable(
                                (mesh.vertices.len() * std::mem::size_of::<egui::epaint::Vertex>())
                                    as u64,
                                vk::BufferUsageFlags::VERTEX_BUFFER,
                            ))
                            .unwrap();
                        Buffer::copy_from_slice(
                            buf.get_mut().unwrap(),
                            0,
                            cast_slice(&mesh.vertices),
                        );
                        buf
                    };
                    let vert_buf = render_graph.bind_node(vert_buf);

                    trace!("verts: {:#?}", mesh.vertices[0]);

                    #[repr(C)]
                    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                    struct PushConstants {
                        screen_size: [f32; 2],
                    }

                    let push_constants = PushConstants {
                        screen_size: [
                            target_info.width as f32 / self.egui_ctx.pixels_per_point(),
                            target_info.height as f32 / self.egui_ctx.pixels_per_point(),
                        ],
                    };

                    let num_indices = mesh.indices.len() as u32;

                    render_graph
                        .begin_pass("Egui pass")
                        .bind_pipeline(&self.ppl)
                        .access_node(idx_buf, AccessType::IndexBuffer)
                        .access_node(vert_buf, AccessType::VertexBuffer)
                        .access_descriptor((0, 0), *texture, AccessType::FragmentShaderReadOther)
                        .load_color(0, target)
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
                }
                _ => panic!("Primitiv callback not yet supported."),
            }
        }
    }

    pub fn run(
        &mut self,
        window: &Window,
        events: &[Event<()>],
        target: impl Into<AnyImageNode>,
        render_graph: &mut RenderGraph,
        ui_fn: impl FnMut(&egui::Context),
    ) {
        // Update events and generate shapes and texture deltas.
        for event in events {
            match event {
                Event::WindowEvent { event, .. } => {
                    self.egui_winit.on_event(&self.egui_ctx, event);
                }
                _ => {}
            }
        }
        let raw_input = self.egui_winit.take_egui_input(window);
        let full_output = self.egui_ctx.run(raw_input, ui_fn);

        self.egui_winit
            .handle_platform_output(window, &self.egui_ctx, full_output.platform_output);

        let deltas = full_output.textures_delta;

        let bound_tex = self.bind_and_update_textures(&deltas, render_graph);

        self.draw_primitive(full_output.shapes, &bound_tex, render_graph, target);

        self.unbind_and_free(bound_tex, render_graph, &deltas);
    }
}
