#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf/transform_datatypes.h>

namespace point_lio_unilidar
{

class OccupancyGridMappingNode
{
public:
    OccupancyGridMappingNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
        loadParameters();
        initialiseMap();

        cloud_sub_ = nh_.subscribe(point_cloud_topic_, 5, &OccupancyGridMappingNode::pointCloudCallback, this);
        odom_sub_ = nh_.subscribe(odom_topic_, 50, &OccupancyGridMappingNode::odomCallback, this);
        map_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 1, true);

        if (publish_frequency_ > 0.0)
        {
            publish_timer_ = nh_.createTimer(ros::Duration(1.0 / publish_frequency_), &OccupancyGridMappingNode::publishTimerCallback, this);
        }
    }

private:
    void loadParameters()
    {
        pnh_.param<std::string>("point_cloud_topic", point_cloud_topic_, std::string("/pointlio/cloud_registered"));
        pnh_.param<std::string>("odom_topic", odom_topic_, std::string("/pointlio/odom"));
        pnh_.param<std::string>("map_frame", map_frame_, std::string("camera_init"));

        pnh_.param<double>("resolution", resolution_, 0.2);
        pnh_.param<int>("width", width_, 400);
        pnh_.param<int>("height", height_, 400);
        pnh_.param<double>("origin_x", origin_x_, -width_ * resolution_ * 0.5);
        pnh_.param<double>("origin_y", origin_y_, -height_ * resolution_ * 0.5);

        pnh_.param<double>("min_height", min_height_, -1.0);
        pnh_.param<double>("max_height", max_height_, 2.0);
        pnh_.param<double>("max_range", max_range_, 80.0);

        pnh_.param<double>("hit_probability", hit_probability_, 0.7);
        pnh_.param<double>("miss_probability", miss_probability_, 0.4);
        pnh_.param<double>("occupied_threshold", occupied_threshold_, 0.65);
        pnh_.param<double>("free_threshold", free_threshold_, 0.35);

        hit_probability_ = std::min(std::max(hit_probability_, 0.51), 0.99);
        miss_probability_ = std::min(std::max(miss_probability_, 0.01), 0.49);

        pnh_.param<double>("publish_frequency", publish_frequency_, 5.0);

        pnh_.param<bool>("use_keyframe", use_keyframe_, false);
        pnh_.param<double>("keyframe_translation", keyframe_translation_thresh_, 0.5);
        pnh_.param<double>("keyframe_rotation", keyframe_rotation_thresh_, 0.35);

        pnh_.param<bool>("publish_on_update", publish_on_update_, false);
        pnh_.param<bool>("continuous_publish", continuous_publish_, false);

        if (publish_frequency_ <= 0.0 && !publish_on_update_)
        {
            ROS_WARN("publish_frequency is non-positive and publish_on_update is false, enabling publish_on_update to avoid missing map updates.");
            publish_on_update_ = true;
        }
    }

    void initialiseMap()
    {
        const std::size_t cell_count = static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
        cells_.assign(cell_count, kUnknownProbabilityValue);
        map_.info.resolution = resolution_;
        map_.info.width = width_;
        map_.info.height = height_;
        map_.info.origin.position.x = origin_x_;
        map_.info.origin.position.y = origin_y_;
        map_.info.origin.position.z = 0.0;
        map_.info.origin.orientation.w = 1.0;
        map_.info.origin.orientation.x = 0.0;
        map_.info.origin.orientation.y = 0.0;
        map_.info.origin.orientation.z = 0.0;
        map_.header.frame_id = map_frame_;
        map_.data.assign(cell_count, -1);

        hit_lookup_table_ = createLookupTableToApplyOdds(probabilityToOdds(static_cast<float>(hit_probability_)));
        miss_lookup_table_ = createLookupTableToApplyOdds(probabilityToOdds(static_cast<float>(miss_probability_)));
    }

    void odomCallback(const nav_msgs::OdometryConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(pose_mutex_);
        latest_pose_ = msg->pose.pose;
        has_odom_ = true;
    }

    bool shouldIntegrate(const geometry_msgs::Pose &pose)
    {
        if (!use_keyframe_)
        {
            return true;
        }

        const double current_yaw = yawFromQuaternion(pose.orientation);
        if (!has_key_pose_)
        {
            last_key_pose_ = pose;
            last_key_yaw_ = current_yaw;
            has_key_pose_ = true;
            return true;
        }

        const double dx = pose.position.x - last_key_pose_.position.x;
        const double dy = pose.position.y - last_key_pose_.position.y;
        const double translation = std::hypot(dx, dy);
        const double rotation = std::fabs(normalizeAngle(current_yaw - last_key_yaw_));

        if (translation >= keyframe_translation_thresh_ || rotation >= keyframe_rotation_thresh_)
        {
            last_key_pose_ = pose;
            last_key_yaw_ = current_yaw;
            return true;
        }

        return false;
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud)
    {
        geometry_msgs::Pose pose;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            if (!has_odom_)
            {
                ROS_WARN_THROTTLE(1.0, "OccupancyGridMapping: waiting for odometry");
                return;
            }
            pose = latest_pose_;
        }

        if (!shouldIntegrate(pose))
        {
            return;
        }

        const double sensor_x = pose.position.x;
        const double sensor_y = pose.position.y;

        int origin_x_cell;
        int origin_y_cell;
        if (!worldToMap(sensor_x, sensor_y, origin_x_cell, origin_y_cell))
        {
            ROS_WARN_THROTTLE(1.0, "Sensor origin is outside of the occupancy grid bounds");
            return;
        }

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud, "z");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            const double px = *iter_x;
            const double py = *iter_y;
            const double pz = *iter_z;

            if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz))
            {
                continue;
            }

            if (pz < min_height_ || pz > max_height_)
            {
                continue;
            }

            const double dx = px - sensor_x;
            const double dy = py - sensor_y;
            const double distance = std::hypot(dx, dy);

            if (max_range_ > 0.0 && distance > max_range_)
            {
                continue;
            }

            int cell_x;
            int cell_y;
            if (!worldToMap(px, py, cell_x, cell_y))
            {
                continue;
            }

            updateFreeCells(origin_x_cell, origin_y_cell, cell_x, cell_y);
            updateOccupiedCell(cell_x, cell_y);
        }

        map_updated_ = true;

        if (publish_on_update_)
        {
            publishMap(ros::Time::now());
        }
    }

    void publishTimerCallback(const ros::TimerEvent &event)
    {
        if (!map_updated_ && !continuous_publish_)
        {
            return;
        }
        publishMap(event.current_real);
    }

    void publishMap(const ros::Time &stamp)
    {
        if (cells_.empty())
        {
            return;
        }

        map_.header.stamp = stamp;
        map_.info.map_load_time = stamp;

        map_.data.resize(cells_.size());
        for (std::size_t idx = 0; idx < cells_.size(); ++idx)
        {
            const std::uint16_t value = cells_[idx];
            if (value == kUnknownProbabilityValue)
            {
                map_.data[idx] = -1;
                continue;
            }

            const double probability = static_cast<double>(valueToProbability(value));
            if (probability >= occupied_threshold_)
            {
                map_.data[idx] = 100;
            }
            else if (probability <= free_threshold_)
            {
                map_.data[idx] = 0;
            }
            else
            {
                map_.data[idx] = -1;
            }
        }

        map_pub_.publish(map_);
        map_updated_ = false;
    }

    bool worldToMap(double wx, double wy, int &mx, int &my) const
    {
        mx = static_cast<int>(std::floor((wx - origin_x_) / resolution_));
        my = static_cast<int>(std::floor((wy - origin_y_) / resolution_));

        if (mx < 0 || my < 0 || mx >= width_ || my >= height_)
        {
            return false;
        }

        return true;
    }

    void updateFreeCells(int x0, int y0, int x1, int y1)
    {
        int dx = std::abs(x1 - x0);
        int dy = std::abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int err = dx - dy;
        int x = x0;
        int y = y0;

        while (true)
        {
            if (x == x1 && y == y1)
            {
                break;
            }

            int e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                x += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y += sy;
            }

            if (x == x1 && y == y1)
            {
                break;
            }

            applyLookupTable(x, y, miss_lookup_table_);
        }
    }

    void updateOccupiedCell(int x, int y)
    {
        applyLookupTable(x, y, hit_lookup_table_);
    }

    void applyLookupTable(int x, int y, const std::array<std::uint16_t, kValueCount> &table)
    {
        if (x < 0 || y < 0 || x >= width_ || y >= height_)
        {
            return;
        }

        const std::size_t index = static_cast<std::size_t>(y) * width_ + static_cast<std::size_t>(x);
        cells_[index] = table[cells_[index]];
    }

    static float clampProbability(float probability)
    {
        return std::min(std::max(probability, kMinProbability), kMaxProbability);
    }

    static float probabilityToOdds(float probability)
    {
        const float clamped = clampProbability(probability);
        return clamped / (1.f - clamped);
    }

    static float oddsToProbability(float odds)
    {
        const float positive_odds = std::max(odds, 1e-6f);
        return positive_odds / (1.f + positive_odds);
    }

    static std::uint16_t probabilityToValue(float probability)
    {
        const float clamped = clampProbability(probability);
        const float scaled = (clamped - kMinProbability) * kValueScale;
        const std::uint16_t value = static_cast<std::uint16_t>(std::lround(scaled)) + 1u;
        return std::min<std::uint16_t>(value, kMaxKnownProbabilityValue);
    }

    static float valueToProbability(std::uint16_t value)
    {
        return valueToProbabilityTable()[value];
    }

    static const std::array<float, kValueCount> &valueToProbabilityTable()
    {
        static const std::array<float, kValueCount> table = []()
        {
            std::array<float, kValueCount> data{};
            data[kUnknownProbabilityValue] = kUnknownProbability;
            for (std::size_t i = 1; i < kValueCount; ++i)
            {
                data[i] = kMinProbability + static_cast<float>(i - 1) / kValueScale;
            }
            return data;
        }();
        return table;
    }

    std::array<std::uint16_t, kValueCount> createLookupTableToApplyOdds(float odds) const
    {
        std::array<std::uint16_t, kValueCount> table{};
        table[kUnknownProbabilityValue] = probabilityToValue(oddsToProbability(odds));
        for (std::size_t i = 1; i < kValueCount; ++i)
        {
            const float current_probability = valueToProbability(static_cast<std::uint16_t>(i));
            const float updated_probability = oddsToProbability(probabilityToOdds(current_probability) * odds);
            table[i] = probabilityToValue(updated_probability);
        }
        return table;
    }

    double yawFromQuaternion(const geometry_msgs::Quaternion &q) const
    {
        tf::Quaternion tf_q(q.x, q.y, q.z, q.w);
        double roll, pitch, yaw;
        tf::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        return yaw;
    }

    static double normalizeAngle(double angle)
    {
        const double two_pi = 2.0 * kPi;
        while (angle > kPi)
        {
            angle -= two_pi;
        }
        while (angle < -kPi)
        {
            angle += two_pi;
        }
        return angle;
    }

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Subscriber cloud_sub_;
    ros::Subscriber odom_sub_;
    ros::Publisher map_pub_;
    ros::Timer publish_timer_;

    nav_msgs::OccupancyGrid map_;
    std::vector<std::uint16_t> cells_;
    std::array<std::uint16_t, kValueCount> hit_lookup_table_{};
    std::array<std::uint16_t, kValueCount> miss_lookup_table_{};

    std::string point_cloud_topic_;
    std::string odom_topic_;
    std::string map_frame_;

    double resolution_ = 0.2;
    int width_ = 400;
    int height_ = 400;
    double origin_x_ = 0.0;
    double origin_y_ = 0.0;
    double min_height_ = -1.0;
    double max_height_ = 2.0;
    double max_range_ = 80.0;
    double hit_probability_ = 0.7;
    double miss_probability_ = 0.4;
    double publish_frequency_ = 5.0;
    double occupied_threshold_ = 0.65;
    double free_threshold_ = 0.35;
    bool use_keyframe_ = false;
    double keyframe_translation_thresh_ = 0.5;
    double keyframe_rotation_thresh_ = 0.35;
    bool publish_on_update_ = false;
    bool continuous_publish_ = false;

    static constexpr double kPi = 3.14159265358979323846;
    static constexpr std::uint16_t kUnknownProbabilityValue = 0u;
    static constexpr std::uint16_t kValueCount = 32768u;
    static constexpr std::uint16_t kMaxKnownProbabilityValue = kValueCount - 1u;
    static constexpr float kMinProbability = 0.1f;
    static constexpr float kMaxProbability = 0.9f;
    static constexpr float kUnknownProbability = 0.5f;
    static constexpr float kValueScale = (kValueCount - 2u) / (kMaxProbability - kMinProbability);

    std::mutex pose_mutex_;
    geometry_msgs::Pose latest_pose_;
    bool has_odom_ = false;
    bool map_updated_ = false;

    geometry_msgs::Pose last_key_pose_;
    double last_key_yaw_ = 0.0;
    bool has_key_pose_ = false;
};

} // namespace point_lio_unilidar

int main(int argc, char **argv)
{
    ros::init(argc, argv, "pointlio_occupancy_grid");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    point_lio_unilidar::OccupancyGridMappingNode node(nh, pnh);
    ros::spin();
    return 0;
}

