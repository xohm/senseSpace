from OpenGL.GL import *
from OpenGL.GLU import *
from .enums import Body34Joint, Body18Joint

# Global variables for floor height stabilization
_floor_height_history = []
_stable_floor_height = None
_floor_height_samples = 30  # Number of samples to keep for moving average

def _person_to_dict(person):
    """Normalize a person entry to a dict with 'skeleton' being a list of joint dicts.
    Accepts either a dataclass `Person` (with .skeleton list of Joint objects) or a dict
    (already serialized). Returns a dict view without modifying the original.
    """
    # If it's already a dict-like object
    if isinstance(person, dict):
        return person

    # Dataclass-style: has attribute 'skeleton'
    if hasattr(person, 'skeleton'):
        sk = []
        for j in person.skeleton:
            # joint may be dataclass with .pos attribute or a dict
            if hasattr(j, 'pos'):
                sk.append({
                    'i': getattr(j, 'i', None),
                    'pos': getattr(j, 'pos'),
                    'ori': getattr(j, 'ori', None) if hasattr(j, 'ori') else None,
                    'conf': getattr(j, 'conf', 0.0)
                })
            else:
                sk.append(j)
        return {'skeleton': sk, 'id': getattr(person, 'id', None), 'confidence': getattr(person, 'confidence', None)}

    # Fallback: try to treat as mapping
    try:
        return dict(person)
    except Exception:
        return {'skeleton': []}


class SkeletonVisualizer3D:
    """
    Utility to visualize a Person (from protocol.py) using OpenGL.
    """
    def __init__(self):
        pass

    def draw_person(self, person, color=(0.2, 0.8, 1.0)):
        """
        Draws a person skeleton using OpenGL.
        :param person: Person object (from protocol.py)
        :param color: RGB tuple for the skeleton color
        """
        glColor3f(*color)
        glPointSize(6.0)
        glBegin(GL_POINTS)
        p = _person_to_dict(person)
        for joint in p.get('skeleton', []):
            pos = joint['pos'] if isinstance(joint, dict) else getattr(joint, 'pos', None)
            if pos is None:
                continue
            glVertex3f(pos["x"], pos["y"], pos["z"])
        glEnd()

    def draw_people(self, people, color=(0.2, 0.8, 1.0)):
        """
        Draws multiple people (list of Person objects)
        """
        for person in people:
            self.draw_person(person, color=color)

    def _normalize_confidence(conf):
        """Map various confidence ranges into 0.0..1.0"""
        try:
            if conf is None:
                return 1.0
            val = float(conf)
            # if value appears to be a percentage e.g. 0..100
            if val > 1.0:
                val = val / 100.0
            # clamp
            return max(0.0, min(1.0, val))
        except Exception:
            return 1.0


    def draw_skeleton_with_bones(self, people_data, joint_color=(0.2, 0.8, 1.0), bone_color=(0.8, 0.2, 0.2)):
        """
        Draws skeleton with bone connections for BODY_34 model.
        :param people_data: List of people data (serialized format)
        :param joint_color: RGB tuple for joint points
        :param bone_color: RGB tuple for bone lines
        """

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for p in people_data:
            p = _person_to_dict(p)
            skeleton = p.get("skeleton", [])
            confidence = p.get("confidence", 1.0)
            alpha = type(self)._normalize_confidence(confidence)
            
            # Draw skeleton bones (lines between joints) for BODY_34
            glColor3f(*bone_color)
            glColor4f(bone_color[0], bone_color[1], bone_color[2], alpha)

            glLineWidth(2.0)
            glBegin(GL_LINES)
            
            # BODY_34 bone connections using enum values for clarity
            J = Body34Joint  # Shorthand for readability
            bones = [
                # Spine
                (J.PELVIS.value, J.NAVAL_SPINE.value),
                (J.NAVAL_SPINE.value, J.CHEST_SPINE.value),
                (J.CHEST_SPINE.value, J.NECK.value),
                (J.NECK.value, J.HEAD.value),
                
                # Left arm
                (J.NECK.value, J.LEFT_CLAVICLE.value),
                (J.LEFT_CLAVICLE.value, J.LEFT_SHOULDER.value),
                (J.LEFT_SHOULDER.value, J.LEFT_ELBOW.value),
                (J.LEFT_ELBOW.value, J.LEFT_WRIST.value),
                (J.LEFT_WRIST.value, J.LEFT_HAND.value),
                (J.LEFT_HAND.value, J.LEFT_HANDTIP.value),
                (J.LEFT_HAND.value, J.LEFT_THUMB.value),
                
                # Right arm  
                (J.NECK.value, J.RIGHT_CLAVICLE.value),
                (J.RIGHT_CLAVICLE.value, J.RIGHT_SHOULDER.value),
                (J.RIGHT_SHOULDER.value, J.RIGHT_ELBOW.value),
                (J.RIGHT_ELBOW.value, J.RIGHT_WRIST.value),
                (J.RIGHT_WRIST.value, J.RIGHT_HAND.value),
                (J.RIGHT_HAND.value, J.RIGHT_HANDTIP.value),
                (J.RIGHT_HAND.value, J.RIGHT_THUMB.value),
                
                # Left leg
                (J.PELVIS.value, J.LEFT_HIP.value),
                (J.LEFT_HIP.value, J.LEFT_KNEE.value),
                (J.LEFT_KNEE.value, J.LEFT_ANKLE.value),
                (J.LEFT_ANKLE.value, J.LEFT_FOOT.value),
                (J.LEFT_ANKLE.value, J.LEFT_HEEL.value),
                
                # Right leg
                (J.PELVIS.value, J.RIGHT_HIP.value),
                (J.RIGHT_HIP.value, J.RIGHT_KNEE.value),
                (J.RIGHT_KNEE.value, J.RIGHT_ANKLE.value),
                (J.RIGHT_ANKLE.value, J.RIGHT_FOOT.value),
                (J.RIGHT_ANKLE.value, J.RIGHT_HEEL.value),
                
                # Face
                (J.HEAD.value, J.NOSE.value),
                (J.HEAD.value, J.LEFT_EYE.value),
                (J.HEAD.value, J.RIGHT_EYE.value),
                (J.LEFT_EYE.value, J.LEFT_EAR.value),
                (J.RIGHT_EYE.value, J.RIGHT_EAR.value),
            ]
            
            for bone in bones:
                if bone[0] < len(skeleton) and bone[1] < len(skeleton):
                    j1 = skeleton[bone[0]]["pos"]
                    j2 = skeleton[bone[1]]["pos"]
                    # Handle both Position object and dict
                    if hasattr(j1, 'x'):
                        glVertex3f(j1.x, j1.y, j1.z)
                    else:
                        glVertex3f(j1["x"], j1["y"], j1["z"])
                    if hasattr(j2, 'x'):
                        glVertex3f(j2.x, j2.y, j2.z)
                    else:
                        glVertex3f(j2["x"], j2["y"], j2["z"])
            
            glEnd()
            
            # Draw joint points on top of bones
            glColor3f(*joint_color)
            glPointSize(8.0)
            glBegin(GL_POINTS)
            for j in skeleton:
                pos = j['pos'] if isinstance(j, dict) else getattr(j, 'pos', None)
                if pos is None:
                    continue
                # Handle both Position object and dict
                if hasattr(pos, 'x'):
                    glVertex3f(pos.x, pos.y, pos.z)
                else:
                    glVertex3f(pos["x"], pos["y"], pos["z"])
            glEnd()

        glDisable(GL_BLEND)            


def draw_skeleton(person, color=(0.2, 0.8, 1.0)):
    """
    Backwards-compatible helper that draws a single person's skeleton using the module-level visualizer.
    """
    v = SkeletonVisualizer3D()
    v.draw_person(person, color=color)


def draw_skeletons_with_bones(people_data, joint_color=(0.2, 0.8, 1.0), bone_color=(0.8, 0.2, 0.2)):
    """
    Helper function to draw skeletons with bone connections.
    """
    v = SkeletonVisualizer3D()
    v.draw_skeleton_with_bones(people_data, joint_color=joint_color, bone_color=bone_color)


def estimate_floor_height(people_data):
    """
    Estimate floor height based on foot/ankle positions of tracked people.
    Returns the estimated floor Y coordinate, or None if no data available.
    """
    if not people_data:
        return None
    
    # Collect foot/ankle Y positions
    foot_heights = []
    
    for person in people_data:
        p = _person_to_dict(person)
        skeleton = p.get("skeleton", [])
        if len(skeleton) > 32:  # Ensure we have BODY_34 data with heel joints
            # Get foot/ankle joint indices (BODY_34 model) - using correct indices
            ankle_joints = [20, 24]  # LEFT_ANKLE, RIGHT_ANKLE
            foot_joints = [21, 25]   # LEFT_FOOT, RIGHT_FOOT
            heel_joints = [32, 33]   # LEFT_HEEL, RIGHT_HEEL
            
            for joint_idx in ankle_joints + foot_joints + heel_joints:
                if joint_idx < len(skeleton):
                    joint = skeleton[joint_idx]
                    y_pos = joint["pos"]["y"]
                    foot_heights.append(y_pos)
    
    if foot_heights:
        # Use the minimum Y value (lowest point) as floor estimate
        # Add a small offset as feet aren't exactly on the floor
        estimated_floor = min(foot_heights) - 50  # 50mm below lowest foot point
        return estimated_floor
    
    return None


def get_stable_floor_height(people_data):
    """
    Get a stable floor height that doesn't jump around frame by frame.
    Uses a moving average of floor height estimates.
    """
    global _floor_height_history, _stable_floor_height
    
    # Get current frame estimate
    current_estimate = estimate_floor_height(people_data)
    
    if current_estimate is not None:
        # Add to history
        _floor_height_history.append(current_estimate)
        
        # Keep only recent samples
        if len(_floor_height_history) > _floor_height_samples:
            _floor_height_history.pop(0)
        
        # Calculate stable height using median (more robust than mean)
        if len(_floor_height_history) >= 5:  # Need at least 5 samples
            sorted_heights = sorted(_floor_height_history)
            median_height = sorted_heights[len(sorted_heights) // 2]
            
            # Only update if the change is significant (> 50mm) to avoid drift
            if _stable_floor_height is None or abs(median_height - _stable_floor_height) > 50:
                _stable_floor_height = median_height
    
    return _stable_floor_height


def draw_floor_grid(size=2000, spacing=100, height=0, color=(0.3, 0.3, 0.3), people_data=None, zed_floor_height=None):
    """
    Draw a floor grid for spatial reference.
    :param size: Total size of the grid in mm (ZED coordinates)
    :param spacing: Spacing between grid lines in mm
    :param height: Y-coordinate height of the floor (None for auto-detection)
    :param color: RGB tuple for grid line color
    :param people_data: People data for floor height estimation (if height=None)
    :param zed_floor_height: Floor height detected by ZED SDK findFloorPlane (preferred)
    """
    # Priority order for floor height detection:
    # 1. Explicit height parameter
    # 2. ZED SDK detected floor height
    # 3. Stable foot-based estimation from people data
    # 4. Fallback to origin (0)
    
    glColor3f(*color)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    
    # Calculate grid bounds
    half_size = size // 2
    
    # Draw lines parallel to X-axis (running along Z)
    for z in range(-half_size, half_size + 1, spacing):
        glVertex3f(-half_size, height, z)
        glVertex3f(half_size, height, z)
    
    # Draw lines parallel to Z-axis (running along X)
    for x in range(-half_size, half_size + 1, spacing):
        glVertex3f(x, height, -half_size)
        glVertex3f(x, height, half_size)
    
    glEnd()
    
    # Draw coordinate axes for reference
    glLineWidth(3.0)
    glBegin(GL_LINES)
    
    # X-axis (red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(0, height, 0)
    glVertex3f(200, height, 0)
    
    # Z-axis (blue) 
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(0, height, 0)
    glVertex3f(0, height, 200)
    
    # Y-axis (green) - pointing up from floor
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0, height, 0)
    glVertex3f(0, height + 200, 0)
    
    glEnd()


def draw_camera(position, orientation, up=(0.0, 1.0, 0.0), fov_deg=45.0, aspect=16.0/9.0, near=200.0, far=800.0, color=(1.0, 1.0, 0.0), scale=1.0, flip=False):
    """
    Draw a simple camera frustum and axes using lines.
    :param position: (x,y,z) or {'x':..,'y':..,'z':..} camera position in world coordinates
    :param orientation: {'x':..,'y':..,'z':..,'w':..} quaternion representing camera rotation
    :param up: up vector (used as fallback)
    :param fov_deg: vertical field of view in degrees
    :param aspect: aspect ratio (width/height)
    :param near: near plane distance from camera
    :param far: far plane distance from camera
    :param color: RGB tuple for camera lines
    :param scale: scale multiplier for visualization size
    :param flip: flip the forward direction
    """
    # Convert position to tuple if it's a dict or Position object
    if isinstance(position, dict):
        pos = (position['x'], position['y'], position['z'])
    elif hasattr(position, 'x'):
        pos = (position.x, position.y, position.z)
    else:
        pos = position
    
    # Convert quaternion to rotation matrix and extract basis vectors
    def quaternion_to_matrix(quat):
        """Convert quaternion {'x':..,'y':..,'z':..,'w':..} or Quaternion object to 3x3 rotation matrix"""
        if isinstance(quat, dict):
            x, y, z, w = quat['x'], quat['y'], quat['z'], quat['w']
        elif hasattr(quat, 'x'):
            x, y, z, w = quat.x, quat.y, quat.z, quat.w
        else:
            x, y, z, w = quat[0], quat[1], quat[2], quat[3]
        
        # Normalize quaternion
        import math
        norm = math.sqrt(x*x + y*y + z*z + w*w)
        if norm == 0:
            return [[1,0,0], [0,1,0], [0,0,1]]  # Identity matrix
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        
        # Convert to rotation matrix
        return [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ]
    
    # small vector helpers
    def sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def add(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    def mul(a, s):
        return (a[0]*s, a[1]*s, a[2]*s)
    def norm(v):
        import math
        l = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        if l == 0:
            return (0.0, 0.0, 0.0)
        return (v[0]/l, v[1]/l, v[2]/l)

    # Get rotation matrix from quaternion
    rot_matrix = quaternion_to_matrix(orientation)
    
    # Extract camera basis vectors from rotation matrix
    right = (rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0])    # X-axis
    cam_up = (rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1])   # Y-axis
    forward = (-rot_matrix[0][2], -rot_matrix[1][2], -rot_matrix[2][2])  # -Z-axis (camera forward)
    
    if flip:
        forward = (-forward[0], -forward[1], -forward[2])

    import math
    fov_rad = math.radians(fov_deg)
    nh = math.tan(fov_rad / 2.0) * near
    nw = nh * aspect
    fh = math.tan(fov_rad / 2.0) * far
    fw = fh * aspect

    # near/far centers
    nc = add(pos, mul(forward, near*scale))
    fc = add(pos, mul(forward, far*scale))

    # near plane corners
    ntl = add(add(nc, mul(cam_up, nh*scale)), mul(right, -nw*scale))
    ntr = add(add(nc, mul(cam_up, nh*scale)), mul(right, nw*scale))
    nbl = add(add(nc, mul(cam_up, -nh*scale)), mul(right, -nw*scale))
    nbr = add(add(nc, mul(cam_up, -nh*scale)), mul(right, nw*scale))

    # far plane corners
    ftl = add(add(fc, mul(cam_up, fh*scale)), mul(right, -fw*scale))
    ftr = add(add(fc, mul(cam_up, fh*scale)), mul(right, fw*scale))
    fbl = add(add(fc, mul(cam_up, -fh*scale)), mul(right, -fw*scale))
    fbr = add(add(fc, mul(cam_up, -fh*scale)), mul(right, fw*scale))

    glColor3f(*color)
    glLineWidth(1.5)

    # draw frustum lines
    glBegin(GL_LINES)
    # near plane outline
    for a, b in ((ntl, ntr), (ntr, nbr), (nbr, nbl), (nbl, ntl)):
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])

    # far plane outline
    for a, b in ((ftl, ftr), (ftr, fbr), (fbr, fbl), (fbl, ftl)):
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])

    # connect near to far
    for a, b in ((ntl, ftl), (ntr, ftr), (nbl, fbl), (nbr, fbr)):
        glVertex3f(a[0], a[1], a[2])
        glVertex3f(b[0], b[1], b[2])

    # lines from camera position to near plane corners (visualizes camera pyramid)
    for corner in (ntl, ntr, nbl, nbr):
        glVertex3f(pos[0], pos[1], pos[2])
        glVertex3f(corner[0], corner[1], corner[2])

    glEnd()

    # draw small local axes at camera position
    glLineWidth(2.0)
    glBegin(GL_LINES)
    # X (red) - camera right
    glColor3f(1.0, 0.0, 0.0)
    rx = add(pos, mul(right, 50.0*scale))
    glVertex3f(pos[0], pos[1], pos[2])
    glVertex3f(rx[0], rx[1], rx[2])
    # Y (green) - camera up
    glColor3f(0.0, 1.0, 0.0)
    uy = add(pos, mul(cam_up, 50.0*scale))
    glVertex3f(pos[0], pos[1], pos[2])
    glVertex3f(uy[0], uy[1], uy[2])
    # Z (blue) - camera forward
    glColor3f(0.0, 0.0, 1.0)
    fz = add(pos, mul(forward, 100.0*scale))
    glVertex3f(pos[0], pos[1], pos[2])
    glVertex3f(fz[0], fz[1], fz[2])
    glEnd()
