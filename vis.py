        app = gui.Application.instance
        app.initialize()

        points = make_point_cloud(100, (0, 0, 0), 1.0)

        w = app.create_window("Open3D - 3D Text", 1024, 768)
        widget3d = gui.SceneWidget()
        widget3d.scene = rendering.Open3DScene(w.renderer)
        widget3d.frame = gui.Rect(0, w.content_rect.y,
                                        900, w.content_rect.height)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = 5 * w.scaling
        widget3d.scene.add_geometry("Points", points, mat)
        for idx in range(0, len(points.points)):
            l = widget3d.add_3d_label(points.points[idx], "{}".format(idx))
            l.color = gui.Color(points.colors[idx][0], points.colors[idx][1],
                                points.colors[idx][2])
            l.scale = np.random.uniform(0.5, 1)
        # bbox = widget3d.scene.bounding_box
        # widget3d.setup_camera(60.0, bbox, bbox.get_center())
        w.add_child(widget3d)
        
        if True:
            # gui layout
            em = w.theme.font_size
            gui_layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
            # create frame that encapsulates the gui
            gui_layout.frame = gui.Rect(700, w.content_rect.y,
                                        200, w.content_rect.height)
            
            collapse = gui.CollapsableVert("Widgets", 0.33*em,
                                            gui.Margins(em, 0, 0, 0))
            logo = gui.ImageWidget('/workspace/raw_data/open_data/kitti/training/image_2/000000.png')
            collapse.add_child(logo)
            gui_layout.add_child(collapse)
            w.add_child(gui_layout)