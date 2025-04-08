import sys
import bpy


def reset_modules():
    """Resets the modules from the library."""
    to_delete = []
    for modname, _ in sys.modules.items():
        if modname.startswith("goo"):
            to_delete.append(modname)
    for modname in to_delete:
        del sys.modules[modname]


def reset_scene():
    """Resets the scene."""
    bpy.app.handlers.frame_change_pre.clear()
    bpy.app.handlers.frame_change_post.clear()
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250

    if bpy.context.active_object and bpy.context.active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)

    for obj in bpy.data.objects:
        # if obj.type in ["CAMERA", "LIGHT"]:
        #     continue
        bpy.data.objects.remove(obj, do_unlink=True)

    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat, do_unlink=True)

    for col in bpy.data.collections:
        bpy.data.collections.remove(col)
