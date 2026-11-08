#include <algorithm>
#include <string>

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/physics/JointController.hh>
#include <gazebo/physics/Model.hh>

#include <gazebo_ros/node.hpp>

#include <std_msgs/msg/float64.hpp>

namespace gaze_attention_gazebo
{

class EyeYawPlugin : public gazebo::ModelPlugin
{
public:

  void Load(gazebo::physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    this->model_ = model;

    // ============================================================
    // YAW JOINT
    // ============================================================

    std::string joint_name =
      sdf->Get<std::string>("joint_name", "eye_yaw_joint").first;

    std::string ros_topic =
      sdf->Get<std::string>("ros_topic", "/eye/target_angle").first;

    this->min_angle_ =
      sdf->Get<double>("min_angle", -0.7854).first;

    this->max_angle_ =
      sdf->Get<double>("max_angle", 0.7854).first;

    double p_gain =
      sdf->Get<double>("p_gain", 10.0).first;

    double i_gain =
      sdf->Get<double>("i_gain", 0.0).first;

    double d_gain =
      sdf->Get<double>("d_gain", 0.5).first;


    this->joint_ = model->GetJoint(joint_name);

    if (!this->joint_) {
      gzerr
        << "[eye_yaw_plugin] joint '" << joint_name
        << "' not found in model '" << model->GetName()
        << "' -- plugin will not move anything.\n";

      return;
    }


    // ============================================================
    // SHARED JOINT CONTROLLER
    // ============================================================

    this->joint_controller_ = model->GetJointController();


    // ============================================================
    // YAW SETUP
    // ============================================================

    this->joint_scoped_name_ =
      this->joint_->GetScopedName();

    this->joint_controller_->SetPositionPID(
      this->joint_scoped_name_,
      gazebo::common::PID(
        p_gain,
        i_gain,
        d_gain
      )
    );

    this->joint_controller_->SetPositionTarget(
      this->joint_scoped_name_,
      0.0
    );


    // ============================================================
    // ROS NODE
    // ============================================================

    this->ros_node_ =
      gazebo_ros::Node::Get(
        sdf,
        "eye_" + this->model_->GetName()
      );


    // ============================================================
    // YAW SUBSCRIBER
    // ============================================================

    this->target_sub_ =
      this->ros_node_->create_subscription<std_msgs::msg::Float64>(
        ros_topic,
        10,

        [this](const std_msgs::msg::Float64::SharedPtr msg)
        {
          double clamped =
            std::clamp(
              msg->data,
              this->min_angle_,
              this->max_angle_
            );

          this->joint_controller_->SetPositionTarget(
            this->joint_scoped_name_,
            clamped
          );
        }
      );


    RCLCPP_INFO(
      this->ros_node_->get_logger(),

      "eye_yaw_plugin: yaw joint='%s' listening on '%s' "
      "(range=[%.4f, %.4f])",

      joint_name.c_str(),
      ros_topic.c_str(),
      this->min_angle_,
      this->max_angle_
    );


    // ============================================================
    // PITCH JOINT
    // ============================================================

    std::string pitch_joint_name =
      sdf->Get<std::string>(
        "pitch_joint_name",
        "eye_pitch_joint"
      ).first;

    std::string pitch_ros_topic =
      sdf->Get<std::string>(
        "pitch_ros_topic",
        "/eye/pitch_angle"
      ).first;

    this->pitch_min_angle_ =
      sdf->Get<double>(
        "pitch_min_angle",
        -0.5
      ).first;

    this->pitch_max_angle_ =
      sdf->Get<double>(
        "pitch_max_angle",
        0.5
      ).first;


    this->pitch_joint_ =
      model->GetJoint(pitch_joint_name);


    if (!this->pitch_joint_) {

      gzerr
        << "[eye_yaw_plugin] pitch joint '"
        << pitch_joint_name
        << "' not found in model '"
        << model->GetName()
        << "'.\n";

      return;
    }


    // ============================================================
    // PITCH CONTROLLER
    // ============================================================

    this->pitch_joint_scoped_name_ =
      this->pitch_joint_->GetScopedName();


    this->joint_controller_->SetPositionPID(
      this->pitch_joint_scoped_name_,
      gazebo::common::PID(
        p_gain,
        i_gain,
        d_gain
      )
    );


    this->joint_controller_->SetPositionTarget(
      this->pitch_joint_scoped_name_,
      0.0
    );


    // ============================================================
    // PITCH SUBSCRIBER
    // ============================================================

    this->pitch_sub_ =
      this->ros_node_->create_subscription<std_msgs::msg::Float64>(
        pitch_ros_topic,
        10,

        [this](const std_msgs::msg::Float64::SharedPtr msg)
        {
          double clamped =
            std::clamp(
              msg->data,
              this->pitch_min_angle_,
              this->pitch_max_angle_
            );

          this->joint_controller_->SetPositionTarget(
            this->pitch_joint_scoped_name_,
            clamped
          );
        }
      );


    RCLCPP_INFO(
      this->ros_node_->get_logger(),

      "eye_yaw_plugin: pitch joint='%s' listening on '%s' "
      "(range=[%.4f, %.4f])",

      pitch_joint_name.c_str(),
      pitch_ros_topic.c_str(),
      this->pitch_min_angle_,
      this->pitch_max_angle_
    );
  }


private:

  // ============================================================
  // MODEL
  // ============================================================

  gazebo::physics::ModelPtr model_;


  // ============================================================
  // YAW
  // ============================================================

  gazebo::physics::JointPtr joint_;

  std::string joint_scoped_name_;

  double min_angle_ = -0.7854;
  double max_angle_ = 0.7854;

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr
    target_sub_;


  // ============================================================
  // PITCH
  // ============================================================

  gazebo::physics::JointPtr pitch_joint_;

  std::string pitch_joint_scoped_name_;

  double pitch_min_angle_ = -0.5;
  double pitch_max_angle_ = 0.5;

  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr
    pitch_sub_;


  // ============================================================
  // SHARED
  // ============================================================

  gazebo::physics::JointControllerPtr joint_controller_;

  gazebo_ros::Node::SharedPtr ros_node_;
};


GZ_REGISTER_MODEL_PLUGIN(EyeYawPlugin)

}  // namespace gaze_attention_gazebo