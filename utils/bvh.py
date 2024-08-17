import collections.abc
import os
import pickle

import numpy as np
from scipy import interpolate


# Quaternion に変換される
# 親子関係が変わっていなければ，boneの定義の順番はどうでも良い

class Vector:
    __slots__ = 'x', 'y', 'z', 'order'

    def __init__(self, values=None, order="xyz"):
        self.order = order
        if values is not None:
            self.set_table(values)
        else:
            self.set_table([np.nan for _ in order])

    def set_table(self, values):
        for key, value in zip(self.order, values):
            setattr(self, key, value)

    def __iter__(self):
        for key in self.order:
            yield getattr(self, key)

    def get_table(self):
        return np.array(list(self))

    def __repr__(self):
        return " ".join(f"{key}:{getattr(self, key)}" for key in self.order)

    def __len__(self):
        i = 0
        for value in self:
            if isinstance(value, collections.abc.Iterable) or not np.isnan(value):
                i += 1
        return i


class Bone:
    __slots__ = 'child_bones', 'name', 'offset', 'parent_bone', 'position', 'rotation'

    def __init__(self, name=None, offset=None, parent_bone=None, child_bones=None, position=None, rotation=None):
        self.offset = Vector(offset, order="xyz")
        self.position = Vector(position, order="xyz")
        self.rotation = Vector(rotation, order="zxy")
        self.child_bones = child_bones or []
        self.name = name or ''
        self.parent_bone = parent_bone

    def kind(self):
        if self.parent_bone is None:
            return 'ROOT'
        elif self.child_bones:
            return 'JOINT'
        return 'End Site'

    # def __del__(self):
    #     if self.parent_bone:
    #         if self.child_bones:
    #             index = self.parent_bone.child_bones.index(self)
    #             del self.parent_bone.child_bones[index]
    #             for i, child in enumerate(self.child_bones, index):
    #                 self.parent_bone.child_bones.insert(i, child)
    #         else:
    #             self.parent_bone.child_bones.remove(self)
    #
    #     for child in self.child_bones:
    #         child.parent_bone = self.parent_bone

    def append(self, bone, index=-1):
        if index == -1:
            self.child_bones.append(bone)
        else:
            self.child_bones.insert(index, bone)
        bone.parent_bone = self

    def delete(self, bone):
        index = self.child_bones.index(bone)
        del self.child_bones[index]  # 追加!!!
        bone.parent_bone = None
        return index

    # def prepend(self, bone):
    #     self.parent_bone = bone
    #     bone.child_bones.append(self)
    #     bone.parent_bone = self.parent_bone
    #     if self.parent_bone:
    #         index = self.parent_bone.child_bones.index(self)
    #         del self.parent_bone.child_bones[index]
    #         self.parent_bone.child_bones.insert(index, bone)

    def __repr__(self):
        return f'{self.name} {self.offset}'


class BVH:
    __slots__ = 'root_bone', 'frame_time', '_gen'

    def __init__(self, file_path, cache=False):
        if cache:
            dirname, filename = os.path.split(file_path)
            cache_dir = os.path.join(dirname, '.bvh_cache')
            cache_path = os.path.join(cache_dir, os.path.splitext(filename)[0] + '.pickle')
            if os.path.isfile(cache_path):
                with open(cache_path, 'rb') as lines:
                    self.root_bone, self.frame_time = pickle.load(lines)
                return

        with open(file_path) as f:
            lines = iter(f)
            next(lines)  # HIERARCHYを取り出す
            self.root_bone = self.parse_bone(lines)

            num_frames = None
            self.frame_time = None
            for line in lines:
                if line.startswith('Frames:'):
                    # num_frames = int(line.removeprefix("Frames:"))
                    num_frames = int(line[7:])
                elif line.startswith('Frame Time:'):
                    # self.frame_time = float(line.removeprefix('Frame Time:'))
                    self.frame_time = float(line[11:])
                if self.frame_time and num_frames:
                    break

            # self.motion_table = np.loadtxt(lines, max_rows=num_frames, dtype=np.float32)
            self.motion_table = np.loadtxt(lines, max_rows=num_frames).T

        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as lines:
                pickle.dump((self.root_bone, self.frame_time), lines)

    @staticmethod
    def parse_bone(lines, current_bone=None):
        while True:
            line = next(lines).split()
            if line[0] == '}':
                break

            next(lines)  # { を取り出す
            offset = np.loadtxt(lines, usecols=(1, 2, 3), max_rows=1)

            if line[0] == 'End':
                bone = Bone(offset=offset)
            else:
                bone = Bone(line[1], offset)
                next(lines)  # CHANNELS を飛ばす

            if current_bone:
                current_bone.append(BVH.parse_bone(lines, bone))
            else:
                current_bone = bone

        i = 1
        for child_bone in current_bone.child_bones:
            if child_bone.kind() == 'End Site':
                child_bone.name = f'{current_bone.name}_end{i}'
                i += 1

        return current_bone

    def __getattr__(self, item):
        for bone in self:
            if bone.name == item:
                return bone
        raise AttributeError

    def optimize(self, accuracy=1e-5):
        while True:
            is_still = True
            for bone in reversed(self):
                if bone.kind() == "End Site" and np.all(bone.offset.get_table() == 0):
                    bone.parent_bone.child_bones.remove(bone)
                    if not bone.parent_bone.child_bones:
                        bone.parent_bone.name += "_end"
                    is_still = False

                elif bone.kind() == "JOINT" and np.all(bone.offset.get_table() == 0) and np.all(
                        abs(bone.rotation.get_table()) < accuracy):
                    index = bone.parent_bone.child_bones.index(bone)
                    bone.parent_bone.child_bones[index:index + 1] = bone.child_bones
                    for child_bones in bone.child_bones:
                        child_bones.parent_bone = bone.parent_bone
                    is_still = False

                elif len(bone.child_bones) == 1 and 0.99 < np.dot(bone.offset.get_table(),
                                                                  bone.child_bones[0].offset.get_table()) / (
                        np.linalg.norm(bone.offset.get_table()) * np.linalg.norm(
                    bone.child_bones[0].offset.get_table())) < 1.01 and np.all(
                    abs(bone.rotation.get_table()) < accuracy):
                    bone.child_bones[0].offset.set_table(
                        bone.child_bones[0].offset.get_table() + bone.offset.get_table())

                    index = bone.parent_bone.child_bones.index(bone)
                    bone.parent_bone.child_bones[index] = bone.child_bones[0]
                    bone.child_bones[0].parent_bone = bone.parent_bone
                    is_still = False

            if is_still:
                break

    @property
    def motion_table(self):
        tmp = []
        for bone in self:
            if bone.kind() == 'ROOT':
                tmp.append(bone.position.get_table())

            if bone.kind() != 'End Site':
                tmp.append(bone.rotation.get_table())

        return np.concatenate(tmp)

    @motion_table.setter
    def motion_table(self, value):
        self.root_bone.position.set_table(value[:3])
        i = 3
        for node in self:
            if node.kind() != 'End Site':
                node.rotation.set_table(value[i:i + 3])
                i += 3

    @property
    def frames(self):
        return self.motion_table.shape[1]

    @property
    def fps(self):
        return 1.0 / self.frame_time

    @fps.setter
    def fps(self, value):
        self.frame_time = 1.0 / value

    @property
    def duration(self):
        return self.frame_time * self.frames

    def subsample(self, fps, kind='cubic'):
        f = interpolate.interp1d(np.arange(0, self.duration, self.frame_time), self.motion_table, kind=kind, axis=1,
                                 fill_value='extrapolate')
        pre_duration = self.duration
        self.motion_table = f((np.arange(1, self.duration, 1.0 / fps)))
        if self.frames == 0:
            raise Exception('Total frames is too few')
        self.frame_time = pre_duration / self.frames

    # def recur(self, ax, current_bone=None):
    #     if current_bone is None:
    #         current_bone = self.root_bone

    def to_bvh(self, file_path):
        with open(file_path, 'w') as f:
            f.write('HIERARCHY\n')
            self._write_bone(f)
            f.write('MOTION\n'
                    f'Frames: {self.frames}\n'
                    f'Frame Time: {self.frame_time}\n')
            np.savetxt(f, self.motion_table.T, fmt='%g', delimiter=' ')

    def to_naoqi(self, file_path, method=1):
        conb = [("LHipRoll", self.LeftUpLeg.rotation.z + 30 + self.LHipJoint.rotation.z),  # 50 は 脚の広げ具合
                ("LHipPitch", self.LeftUpLeg.rotation.x + self.LHipJoint.rotation.x - self.LowerBack.rotation.x),
                ("LKneePitch", self.LeftLeg.rotation.x),
                ("LAnkleRoll", self.LeftFoot.rotation.z),
                ("LAnklePitch", self.LeftFoot.rotation.x),
                ("RHipRoll", self.RightUpLeg.rotation.z - 30 + self.RHipJoint.rotation.z),  # -50 は 脚の広げ具合
                ("RHipPitch", self.RightUpLeg.rotation.x + self.RHipJoint.rotation.x - self.LowerBack.rotation.x),
                ("RKneePitch", self.RightLeg.rotation.x),
                ("RAnkleRoll", self.RightFoot.rotation.z),
                ("RAnklePitch", self.RightFoot.rotation.x),
                ("HeadPitch", self.Neck1.rotation.x),
                ("HeadYaw", self.Neck1.rotation.y),

                # 太ももの回転は左右の脚が連動している
                ("LHipYawPitch", (self.LeftUpLeg.rotation.y + self.LHipJoint.rotation.y) * -1),

                ("LElbowRoll", self.LeftForeArm.rotation.y),
                ("RElbowRoll", self.RightForeArm.rotation.y),
                ("LWristYaw", self.LeftHand.rotation.x),
                ("RWristYaw", self.RightHand.rotation.x * -1)]

        if method == 1:
            conb += [("LShoulderRoll", self.LeftArm.rotation.y),
                     ("LShoulderPitch", self.LeftArm.rotation.x),
                     ("LElbowYaw", self.LeftForeArm.rotation.x),
                     ("RShoulderRoll", self.RightArm.rotation.y),
                     ("RShoulderPitch", self.RightArm.rotation.x),
                     ("RElbowYaw", self.RightForeArm.rotation.x * -1)]
        elif method == 2:
            conb += [("LShoulderRoll", self.LeftArm.rotation.z + 90 + self.LeftShoulder.rotation.z),
                     ("LShoulderPitch", self.LeftArm.rotation.x + 90),
                     ("LElbowYaw", self.LeftForeArm.rotation.x - 90),
                     ("RShoulderRoll", self.RightArm.rotation.z - 90 + self.RightShoulder.rotation.z),
                     ("RShoulderPitch", self.RightArm.rotation.x + 90),
                     ("RElbowYaw", self.RightForeArm.rotation.x * -1 + 90)]

        frames = np.linspace(0, self.duration, self.frames) + 3

        angle_table = np.radians(np.array([val for _, val in conb]))
        table = np.vstack([frames, angle_table])
        np.savetxt(file_path, table.T, fmt='%g', delimiter=', ', header=f"frames, {', '.join(key for key, _ in conb)}",
                   comments='')

    def _write_bone(self, file_obj, bone=None, indent=0):
        if bone is None:
            bone = self.root_bone

        pre_indent = '\t' * indent
        kind_of_bone = bone.kind()

        file_obj.write(f'{pre_indent}{kind_of_bone}{"" if kind_of_bone == "End Site" else f" {bone.name}"}\n'
                       f'{pre_indent}{{\n'
                       f'{pre_indent}\tOFFSET ')

        np.savetxt(file_obj, bone.offset.get_table(), fmt='%g', delimiter=' ', newline=' ')
        file_obj.write('\n')

        if kind_of_bone == 'ROOT':
            file_obj.write(f'{pre_indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n')
        elif kind_of_bone == 'JOINT':
            file_obj.write(f'{pre_indent}\tCHANNELS 3 Zrotation Xrotation Yrotation\n')

        for child_bone in bone.child_bones:
            self._write_bone(file_obj, child_bone, indent + 1)

        file_obj.write(f'{pre_indent}}}\n')

    def __iter__(self):
        return self._iter()

    def __reversed__(self):
        return self._reversed()

    def _iter(self, node=None):
        if node is None:
            node = self.root_bone
        yield node
        for n in node.child_bones:
            yield from self._iter(n)

    def _reversed(self, current_bone=None):
        if current_bone is None:
            current_bone = self.root_bone

        for bone in reversed(current_bone.child_bones):
            yield from self._reversed(bone)
        yield current_bone

    def __len__(self):
        i = 0
        for _ in self:
            i += 1
        return i

    def __getitem__(self, item):
        if isinstance(item, str):
            for bone in self:
                if bone.name == item:
                    return bone
            raise KeyError
        elif isinstance(item, int):
            if item >= 0:
                for i, bone in enumerate(self):
                    if i == item:
                        return bone
            else:
                item *= -1
                for i, bone in enumerate(reversed(self), 1):
                    if i == item:
                        return bone
            raise IndexError
        raise TypeError

    def __repr__(self, bone=None, indent=0):
        if bone is None:
            bone = self.root_bone

        repr_ = '  ' * indent + repr(bone) + '\n'

        for child_bone in bone.child_bones:
            repr_ += self.__repr__(child_bone, indent + 1)

        return repr_
