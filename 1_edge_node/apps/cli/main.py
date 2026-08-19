import argparse
import sys

def main():
    """
    Entrypoint cho giao diện dòng lệnh hợp nhất (Unified CLI).
    Giống như lệnh `yolo` của Ultralytics, CLI này bọc toàn bộ platform.
    """
    parser = argparse.ArgumentParser(
        prog="vision-cli",
        description="Giao diện dòng lệnh cho Vision AI Platform."
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Các lệnh khả dụng")
    
    # Lệnh chạy Edge Agent
    run_parser = subparsers.add_parser("run", help="Chạy một ứng dụng")
    run_parser.add_parser("edge-agent", help="Khởi động Edge Agent Node")
    
    # Lệnh train model
    train_parser = subparsers.add_parser("train", help="Huấn luyện mô hình AI")
    train_parser.add_argument("--data", type=str, help="Đường dẫn đến file dataset config")
    train_parser.add_argument("--model", type=str, help="Tên model base")
    
    # Lệnh export
    export_parser = subparsers.add_parser("export", help="Xuất mô hình sang định dạng khác (ONNX, TensorRT)")
    export_parser.add_argument("--weights", type=str, help="Đường dẫn file weights")
    export_parser.add_argument("--format", type=str, choices=["onnx", "engine", "openvino"])

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    print(f"🚀 [vision-cli] Đang thực thi lệnh: {args.command}")
    
    if args.command == "run":
        # Ví dụ gọi hàm main của web_backend
        print("-> Khởi động ứng dụng Edge Agent...")
        # from apps.edge_agent.web_backend import start_server
        # start_server()
        
    elif args.command == "train":
        print(f"-> Bắt đầu huấn luyện mô hình {args.model} với data {args.data}...")
        
    elif args.command == "export":
        print(f"-> Đang xuất weights từ {args.weights} sang định dạng {args.format}...")

if __name__ == "__main__":
    main()
