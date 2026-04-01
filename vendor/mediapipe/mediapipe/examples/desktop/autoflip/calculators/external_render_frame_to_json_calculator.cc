#include <iomanip>
#include <sstream>
#include <string>

#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {
namespace autoflip {

constexpr char kInputTag[] = "EXTERNAL_RENDER_FRAME";
constexpr char kOutputTag[] = "JSON";

class ExternalRenderFrameToJsonCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kInputTag).Set<ExternalRenderFrame>();
    cc->Outputs().Tag(kOutputTag).Set<std::string>();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& frame =
        cc->Inputs().Tag(kInputTag).Get<ExternalRenderFrame>();
    RET_CHECK(frame.has_normalized_crop_from_location())
        << "normalized_crop_from_location missing";

    const auto& rect = frame.normalized_crop_from_location();
    const float x_center_norm = rect.x() + rect.width() / 2.0f;

    std::ostringstream oss;
    oss << std::setprecision(6);
    oss << "{\"timestamp_us\":" << frame.timestamp_us()
        << ",\"x_center_norm\":" << x_center_norm
        << ",\"crop_x_norm\":" << rect.x()
        << ",\"crop_width_norm\":" << rect.width()
        << "}";

    cc->Outputs()
        .Tag(kOutputTag)
        .AddPacket(MakePacket<std::string>(oss.str()).At(cc->InputTimestamp()));
    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(ExternalRenderFrameToJsonCalculator);

}  // namespace autoflip
}  // namespace mediapipe
