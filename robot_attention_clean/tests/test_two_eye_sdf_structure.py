"""Structural validation for gazebo/worlds/two_eye_demo.sdf's eye redesign
(fixed black iris, rotating white eye shell) -- pure XML parsing, no
Gazebo/ROS required, so it's fast and reproducible in plain CI/pytest.

Covers requirement 19:
  - both intended controlled joints exist (left_eye_yaw_joint, right_eye_yaw_joint)
  - both controller plugin blocks use /gaze_attention/iris_yaw_cmd
  - the black iris links are NOT children of the controlled (revolute) joints
  - both controlled joints' limits contain [-0.8, +0.8] radians
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SDF_PATH = Path(__file__).resolve().parent.parent / "gazebo" / "worlds" / "two_eye_demo.sdf"

REQUIRED_MIN_LIMIT = -0.8
REQUIRED_MAX_LIMIT = 0.8

CONTROLLED_JOINTS = ("left_eye_yaw_joint", "right_eye_yaw_joint")
IRIS_LINKS = ("left_iris_link", "right_iris_link")
COMMAND_TOPIC = "/gaze_attention/iris_yaw_cmd"


@pytest.fixture(scope="module")
def sdf_root():
    assert SDF_PATH.is_file(), f"SDF not found at {SDF_PATH}"
    return ET.parse(SDF_PATH).getroot()


@pytest.fixture(scope="module")
def robot_model(sdf_root):
    model = sdf_root.find(".//model[@name='two_eye_robot']")
    assert model is not None, "model 'two_eye_robot' not found -- model name must be preserved"
    return model


def _joints_by_name(model) -> dict:
    return {j.attrib["name"]: j for j in model.findall("joint")}


def _links_by_name(model) -> dict:
    return {l.attrib["name"]: l for l in model.findall("link")}


# --- both intended controlled joints exist ------------------------------------

def test_both_controlled_eye_yaw_joints_exist(robot_model):
    joints = _joints_by_name(robot_model)
    for name in CONTROLLED_JOINTS:
        assert name in joints, f"controlled joint {name!r} not found"
        assert joints[name].attrib.get("type") == "revolute", f"{name} must be a revolute joint"


def test_controlled_joints_are_not_static(robot_model):
    """The controlled white-eye links must be attached via a real revolute
    joint (not fixed, not a <static> link)."""
    links = _links_by_name(robot_model)
    joints = _joints_by_name(robot_model)
    for joint_name in CONTROLLED_JOINTS:
        child_name = joints[joint_name].find("child").text
        child_link = links.get(child_name)
        assert child_link is not None, f"child link {child_name!r} of {joint_name} not found"
        assert child_link.find("static") is None, f"{child_name} must not be marked <static>"


# --- both controller blocks use the exact command topic, topic unchanged -----

def test_both_controllers_use_the_exact_command_topic(robot_model):
    controllers = [
        p for p in robot_model.findall("plugin")
        if p.attrib.get("name") == "gz::sim::systems::JointPositionController"
    ]
    controlled = [c for c in controllers if c.find("joint_name").text in CONTROLLED_JOINTS]
    assert len(controlled) == 2, (
        f"expected exactly 2 JointPositionController plugins for {CONTROLLED_JOINTS}, found {len(controlled)}"
    )
    for c in controlled:
        topic = c.find("topic")
        assert topic is not None and topic.text == COMMAND_TOPIC, (
            f"controller for joint {c.find('joint_name').text!r} must use topic {COMMAND_TOPIC!r}, "
            f"got {topic.text if topic is not None else None!r}"
        )


def test_no_stray_controllers_on_other_joints(robot_model):
    """Only the two intended eye-yaw joints should be driven by a
    JointPositionController -- e.g. the iris/base fixed joints must not
    accidentally have one too."""
    controllers = [
        p for p in robot_model.findall("plugin")
        if p.attrib.get("name") == "gz::sim::systems::JointPositionController"
    ]
    controlled_joint_names = {c.find("joint_name").text for c in controllers}
    assert controlled_joint_names == set(CONTROLLED_JOINTS)


# --- black iris links are NOT children of the controlled joints --------------

def test_iris_links_are_not_children_of_controlled_joints(robot_model):
    joints = _joints_by_name(robot_model)
    controlled_children = {joints[name].find("child").text for name in CONTROLLED_JOINTS}
    for iris_link in IRIS_LINKS:
        assert iris_link not in controlled_children, (
            f"{iris_link} must not be the child of a controlled (revolute) joint -- it must stay fixed"
        )


def test_iris_links_are_children_of_fixed_joints_only(robot_model):
    """Every joint that has an iris link as its child must be type 'fixed'."""
    for joint in robot_model.findall("joint"):
        child = joint.find("child")
        if child is not None and child.text in IRIS_LINKS:
            assert joint.attrib.get("type") == "fixed", (
                f"joint {joint.attrib.get('name')!r} parents iris link {child.text!r} "
                f"but is type {joint.attrib.get('type')!r}, expected 'fixed'"
            )


def test_iris_links_have_no_revolute_joint_at_all(robot_model):
    """No joint of type 'revolute' may have an iris link as its child,
    regardless of joint name -- the iris must never be the moving part."""
    for joint in robot_model.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            continue
        child = joint.find("child")
        assert child is None or child.text not in IRIS_LINKS, (
            f"revolute joint {joint.attrib.get('name')!r} must not drive an iris link"
        )


# --- joint limits contain [-0.8, +0.8] radians --------------------------------

def test_controlled_joint_limits_contain_required_range(robot_model):
    joints = _joints_by_name(robot_model)
    for name in CONTROLLED_JOINTS:
        limit = joints[name].find("axis/limit")
        assert limit is not None, f"{name} has no <axis><limit> element"
        lower = float(limit.find("lower").text)
        upper = float(limit.find("upper").text)
        assert lower <= REQUIRED_MIN_LIMIT, f"{name} lower limit {lower} does not cover {REQUIRED_MIN_LIMIT}"
        assert upper >= REQUIRED_MAX_LIMIT, f"{name} upper limit {upper} does not cover {REQUIRED_MAX_LIMIT}"


def test_controlled_joint_axis_is_z_yaw(robot_model):
    """Both controlled joints must rotate about the same axis (z, yaw) that
    the previous iris joints used -- required so the existing angle sign
    convention carries over unchanged (requirement: no silent sign flip)."""
    joints = _joints_by_name(robot_model)
    for name in CONTROLLED_JOINTS:
        xyz = joints[name].find("axis/xyz").text.split()
        xyz = [float(v) for v in xyz]
        assert xyz == [0.0, 0.0, 1.0], f"{name} axis must be 0 0 1 (z/yaw), got {xyz}"


# --- structural sanity: model/world names preserved ---------------------------

def test_world_and_model_names_preserved(sdf_root, robot_model):
    world = sdf_root.find("world")
    assert world.attrib.get("name") == "two_eye_demo"
    assert robot_model.attrib.get("name") == "two_eye_robot"


def test_sdf_is_well_formed_xml():
    # A parse failure here would raise before this test even runs (the
    # module-scoped fixture already parses it) -- this test exists so a
    # malformed-SDF failure shows up as a clearly-named test rather than a
    # collection error.
    ET.parse(SDF_PATH)
