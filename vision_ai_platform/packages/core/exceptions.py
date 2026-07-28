from typing import Optional, Dict, Any

class CoreException(Exception):
    """
    Ngoại lệ cơ sở cho toàn bộ hệ thống Vision AI.
    Hỗ trợ Error Code và Metadata để dễ dàng log/trả về API.
    """
    def __init__(
        self, 
        message: str, 
        error_code: str = "INTERNAL_ERROR", 
        payload: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.payload = payload or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert lỗi sang dạng dict để phục vụ Logging JSON hoặc API Response."""
        return {
            "error": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "payload": self.payload
        }


# =====================================================================
# TẦNG 3: HARDWARE, CAMERA & STREAM EXCEPTIONS
# =====================================================================
class HardwareException(CoreException):
    """Lỗi chung cấp phần cứng."""
    pass

class CameraConnectError(HardwareException):
    """Không thể kết nối hoặc mất kết nối tới Camera (RTSP/GigE/USB)."""
    pass

class CameraStreamTimeoutError(HardwareException):
    """Camera ngưng gửi frame quá thời gian quy định (Timeout)."""
    pass

class FrameBufferOverflowError(HardwareException):
    """Tràn bộ nhớ đệm Ring-buffer / Shared Memory / CUDA IPC."""
    pass

class PLCCommunicationError(HardwareException):
    """Lỗi giao tiếp Modbus/Profinet với PLC hoặc thiết bị ngoại vi."""
    pass


# =====================================================================
# TẦNG 4: AI ENGINES & MLOPS EXCEPTIONS
# =====================================================================
class AIException(CoreException):
    """Lỗi cơ sở cho gói packages/ai."""
    pass

class ModelException(AIException):
    """Lỗi liên quan đến nạp hoặc quản lý mô hình."""
    pass

class ModelNotFoundError(ModelException):
    """Không tìm thấy file trọng số (.pt, .onnx, .engine)."""
    pass

class InferenceEngineError(AIException):
    """Lỗi trong quá trình chạy Inference (TensorRT, ONNX Runtime, PyTorch)."""
    pass

class PredictorException(AIException):
    """Lỗi xử lý đầu vào/đầu ra của Predictor."""
    pass

class TrackerException(AIException):
    """Lỗi trong thuật toán theo dõi đối tượng (ByteTrack, SORT...)."""
    pass

class TrainerException(AIException):
    """Lỗi phát sinh trong quá trình huấn luyện (DDP, CUDA OOM...)."""
    pass

class EvaluatorException(AIException):
    """Lỗi tính toán chỉ số mAP, Loss, F1-Score."""
    pass

class ExporterException(AIException):
    """Lỗi khi export mô hình (PyTorch -> ONNX -> TensorRT)."""
    pass

class DatasetValidationError(AIException):
    """Tập dữ liệu không đúng cấu hình YAML, thiếu nhãn hoặc hỏng file ảnh."""
    pass

class DataDriftDetectedError(AIException):
    """Phát hiện suy giảm chất lượng dữ liệu/phân phối ảnh thực tế."""
    pass


# =====================================================================
# TẦNG 5: PIPELINE & WORKFLOW EXCEPTIONS
# =====================================================================
class WorkflowException(CoreException):
    """Lỗi khi khởi tạo hoặc thực thi DAG Workflow."""
    pass

class WorkflowNodeError(WorkflowException):
    """Lỗi xảy ra tại một Node cụ thể trong Pipeline."""
    pass

class WorkflowTimeoutError(WorkflowException):
    """Pipeline thực thi quá thời gian cho phép (SLA Violation)."""
    pass


# =====================================================================
# TẦNG 6: SERVICES & NETWORK EXCEPTIONS
# =====================================================================
class ServiceException(CoreException):
    """Lỗi cấp Microservice/Daemon."""
    pass

class NetworkCommunicationError(ServiceException):
    """Lỗi truyền nhận gRPC, REST API, MQTT hoặc Kafka."""
    pass


# =====================================================================
# TẦNG 7: DOMAIN SOLUTION EXCEPTIONS
# =====================================================================
class SolutionException(CoreException):
    """Lỗi liên quan đến Business Logic theo ngành (Smart Factory, Retail...)."""
    pass