import sys
import os
import logging
import numpy as np
import bpy
import io
from enum import Enum, Flag, auto
from typing import Union, List, Optional
from contextlib import contextmanager

from goo.handler import Handler, StopHandler
from goo.cell import Cell, CellType
from goo.molecule import DiffusionSystem

Render = Enum("Render", ["PNG", "TIFF", "MP4"])


class Simulator:
    """A simulator for cell-based simulations in Blender.

    Args:
        cells (List[Cell]): List of cells.
        time (List[int]): Start and end frames.
        physics_dt (int): Time step for physics simulation.
        molecular_dt (int): Time step for molecular simulation.

    """

    # TODO: determine diffsystem or diffsystems
    def __init__(
        self,
        celltypes: List[Union[CellType, Cell]] = [],
        diffsystems: DiffusionSystem = [],
        time: int = 250,
        physics_dt: int = 1,
        molecular_dt: int = 1,
    ):
        self.celltypes = celltypes
        # takes first possible diffsystem
        self.diffsystem = diffsystems[0] if diffsystems else None
        self.physics_dt = physics_dt
        self.molecular_dt = molecular_dt
        # In Blender 4.5, some addons might be renamed or not available
        self.addons = []  # Removing add_mesh_extra_objects as it's causing issues in 4.5
        self.render_format: Render = Render.PNG
        self.time = time

        # Set up simulation parameters for diffusion system
        if self.diffsystem is not None:
            self.diffsystem.time_step = molecular_dt
            self.diffsystem.total_time = physics_dt

    def set_seed(self, seed):
        """Set the random seed for the simulation."""
        np.random.seed(seed)
        bpy.context.scene["seed"] = seed

    def setup_world(self, seed=1):
        """Set up the Blender scene for the simulation."""
        # Enable addons
        for addon in self.addons:
            self.enable_addon(addon)

        # Set random seed
        self.set_seed(seed)

        # Set up simulation time interval
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = self.time
        
        # Configure Blender's logging to filter dependency messages
        bpy.app.debug_depsgraph = False
        bpy.app.debug_wm = False
        bpy.app.debug = False
        
        # Set units to the metric system
        bpy.context.scene.unit_settings.system = "METRIC"
        bpy.context.scene.unit_settings.scale_length = 1e-6
        bpy.context.scene.unit_settings.system_rotation = "DEGREES"
        bpy.context.scene.unit_settings.length_unit = "MICROMETERS"
        bpy.context.scene.unit_settings.mass_unit = "MILLIGRAMS"
        bpy.context.scene.unit_settings.time_unit = "SECONDS"
        bpy.context.scene.unit_settings.temperature_unit = "CELSIUS"

        # Turn off gravity
        self.toggle_gravity(False)

        # Set up rendering environment
        node_tree = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes
        tree_nodes.clear()

        # Add background node
        node_background = tree_nodes.new(type="ShaderNodeBackground")
        node_environment = tree_nodes.new("ShaderNodeTexEnvironment")
        scripts_paths = bpy.utils.script_paths()
        try:
            node_environment.image = bpy.data.images.load(
                scripts_paths[-1] + "/modules/goo/missile_launch_facility_01_4k.hdr"
            )
        except Exception:
            print(sys.exc_info())
            print(
                """WARNING FROM GOO: To enable proper rendering you must have
                /modules/goo/missile_launch_facility_01_4k.hdr in the right location"""
            )
        node_environment.location = -300, 0

        # Add output node
        node_output = tree_nodes.new(type="ShaderNodeOutputWorld")
        node_output.location = 200, 0
        # Link all nodes
        links = node_tree.links
        links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

        # # set film to transparent to hide background
        bpy.context.scene.render.film_transparent = True

        # Allow cloth physics pass 250 frames
        self.extend_scene()

    def enable_addon(self, addon):
        """Enable an addon in Blender."""
        try:
            if addon not in bpy.context.preferences.addons:
                bpy.ops.preferences.addon_enable(module=addon)
                print(f"Addon '{addon}' has been enabled.")
            else:
                print(f"Addon '{addon}' is already enabled.")
        except Exception as e:
            print(f"Warning: Could not enable addon '{addon}'. Error: {e}")
            # Continue execution even if addon fails to load
            pass

    def toggle_gravity(self, on):
        """Toggle gravity in the scene."""
        bpy.context.scene.use_gravity = on

    def get_cells_func(self, celltypes=None):
        """Get a function that returns all cells in the simulation."""
        celltypes = celltypes if celltypes else self.celltypes

        def get_cells():
            return [cell for celltype in celltypes for cell in celltype.cells]

        return get_cells

    def get_diffsystem_func(self, diffsystem=None):
        """Get a function that returns the diffusion system."""
        diffsystem = diffsystem if diffsystem is not None else self.diffsystem

        def get_diffsystem():
            return diffsystem

        return get_diffsystem

    def get_cells(self, celltypes=None):
        """Get all cells in the simulation."""
        celltypes = celltypes if celltypes else self.celltypes
        return [cell for celltype in celltypes for cell in celltype.cells]

    def extend_scene(self):
        """Extend the scene to allow cloth physics to pass the default 250 frames."""
        cells = self.get_cells()
        for cell in cells:
            if hasattr(cell, 'cloth_mod') and cell.cloth_mod and hasattr(cell.cloth_mod, 'point_cache'):
                if cell.cloth_mod.point_cache.frame_end < self.time:
                    cell.cloth_mod.point_cache.frame_end = self.time

    def add_handler(
        self,
        handler: Handler,
        celltypes: list[CellType] = None,
        diffsystem: DiffusionSystem = None,
    ):
        """Add a handler to the simulation."""
        handler.setup(
            self.get_cells_func(celltypes),
            self.get_diffsystem_func(diffsystem),
            self.physics_dt,
        )

        bpy.app.handlers.frame_change_post.append(handler.run)

    def add_handlers(self, handlers: list[Handler]):
        """Add multiple handlers to the simulation."""
        # always include a stop handler
        stop_handler = StopHandler()
        # Initialize the stop handler with the same parameters as other handlers
        stop_handler.setup(
            self.get_cells_func(),
            self.get_diffsystem_func(),
            self.physics_dt,
        )
        bpy.app.handlers.frame_change_pre.append(stop_handler.run)
        
        for handler in handlers:
            self.add_handler(handler)

    def render(
        self,
        frames: Optional[Union[List[int], range]] = None,
        path: str = None,
        camera=False,
        format: Render = Render.PNG,
    ):
        """
        Render the simulation in the background without
        updating the 3D Viewport in real time.
        If a camera is specified, the frames will be rendered with it,
        otherwise the frames will be rendered in the 3D Viewport.
        It will updated the scene at the end of the simulation.

        Args:
            start (int): Start frame.
            end (int): End frame.
            path (str): Path to save the frames.
            camera (bool): Render with the camera.
            format (Render): Render format: PNG (default), TIFF, MP4.
        """
        if not path:
            path = os.path.dirname(bpy.context.scene.render.filepath)
        else:
            print("Save path not provided. Falling back on default path.")

        print("----- RENDERING... -----")

        render_format = format if format else self.render_format
        match render_format:
            case Render.PNG:
                bpy.context.scene.render.image_settings.file_format = "PNG"
            case Render.TIFF:
                bpy.context.scene.render.image_settings.file_format = "TIFF"
            case Render.MP4:
                bpy.context.scene.render.image_settings.file_format = "FFMPEG"
                bpy.context.scene.render.ffmpeg.format = "MPEG4"

        match frames:
            case None:
                start = 1
                end = self.time
                frame_list = list(range(start, end + 1))
            case range():
                frame_list = list(frames)
            case list():
                frame_list = frames

        for i in range(1, max(frame_list) + 1):
            print(i, end=" ")
            bpy.context.scene.frame_set(i)
            bpy.context.scene.render.filepath = os.path.join(path, f"{i:04d}")
            if i in frame_list:
                if camera:
                    bpy.ops.render.render(write_still=True)
                else:
                    bpy.ops.render.opengl(write_still=True)

        bpy.context.scene.render.filepath = path
        print("\n----- RENDERING COMPLETED! -----")

    def render_animation(self, path=None, end=bpy.context.scene.frame_end, camera=False):
        """Render the simulation as an animation."""
        if not path:
            print("Save path not provided. Falling back on default path.")
            path = os.path.dirname(bpy.context.scene.render.filepath)
        bpy.context.scene.render.filepath = os.path.join(path, "")

        bpy.context.scene.render.image_settings.file_format = "FFMPEG"
        bpy.context.scene.render.ffmpeg.format = "MPEG4"

        print("----- RENDERING... -----")
        print("Rendering to", bpy.context.scene.render.filepath)

        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_set(1)
        bpy.context.scene.frame_end = end
        if camera:
            bpy.ops.render.render(animation=True, write_still=True)
        else:
            bpy.ops.render.opengl(animation=True, write_still=True)
        print("\n----- RENDERING COMPLETED! -----")

    def run(self, end=bpy.context.scene.frame_end):
        """
        Run the simulation in the background without
        updating the 3D Viewport in real time. 

        Args:
            end (int): End frame. Defaults to the last frame of the scene.
        """
        print("----- SIMULATION START -----")
        for i in range(1, end + 1):
            print(i, end=" ")
            bpy.context.scene.frame_set(i)
        print("\n----- SIMULATION END -----")
