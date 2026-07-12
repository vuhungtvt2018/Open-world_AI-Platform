import os
import shutil
import time
import threading

def cleanup_old_sessions(captures_dir: str, days_to_keep: float = 15.0):
    """
    Quét và xoá các thư mục session trong captures/ cũ hơn days_to_keep ngày
    """
    if not os.path.exists(captures_dir):
        print(f"[CLEANUP] Không tìm thấy thư mục captures: {captures_dir}")
        return

    now = time.time()
    cutoff_time = now - (days_to_keep * 86400)  # 86400 giây/ngày

    print(f"[CLEANUP] Bắt đầu quét thư mục: {captures_dir}. Ngưỡng xoá: trước {days_to_keep} ngày.")
    
    # Duyệt qua các thư mục sản phẩm (VD: captures/bulong_8ly/sessions)
    for product_name in os.listdir(captures_dir):
        product_path = os.path.join(captures_dir, product_name)
        if not os.path.isdir(product_path):
            continue
            
        sessions_dir = os.path.join(product_path, "sessions")
        if not os.path.exists(sessions_dir) or not os.path.isdir(sessions_dir):
            continue

        # Duyệt qua từng session
        deleted_count = 0
        for session_name in os.listdir(sessions_dir):
            session_path = os.path.join(sessions_dir, session_name)
            if not os.path.isdir(session_path):
                continue
                
            # Lấy thời gian chỉnh sửa cuối cùng của session (mtime)
            mtime = os.path.getmtime(session_path)
            if mtime < cutoff_time:
                try:
                    shutil.rmtree(session_path)
                    print(f"[CLEANUP] Đã tự động xoá session cũ: {session_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"[CLEANUP ERROR] Lỗi khi xoá {session_path}: {e}")
        
        if deleted_count > 0:
            print(f"[CLEANUP] Đã dọn dẹp xong cho sản phẩm: {product_name}. Xoá thành công {deleted_count} sessions.")

def run_cleaner_daemon(captures_dir: str, days_to_keep: float = 15.0, interval_hours: float = 24.0):
    """
    Khởi chạy background thread: dọn dẹp NGAY LẬP TỨC lúc startup,
    sau đó tự động dọn dẹp định kỳ sau mỗi `interval_hours` giờ.
    """
    def daemon_job():
        # Lần 1: Chạy ngay lập tức khi khởi động hệ thống
        print("[CLEANUP DAEMON] Hệ thống đã khởi động. Tiến hành dọn dẹp ổ cứng lần đầu tiên...")
        try:
            cleanup_old_sessions(captures_dir, days_to_keep)
        except Exception as e:
            print(f"[CLEANUP DAEMON ERROR] Lỗi dọn dẹp startup: {e}")

        # Các lần tiếp theo: Lặp lại sau mỗi `interval_hours` giờ
        while True:
            # Chuyển đổi giờ thành giây
            sleep_time_seconds = interval_hours * 3600
            print(f"[CLEANUP DAEMON] Sẽ tự động dọn dẹp tiếp theo sau {interval_hours} giờ nữa (sleep {sleep_time_seconds}s)...")
            time.sleep(sleep_time_seconds)
            
            print("[CLEANUP DAEMON] Bắt đầu chu kỳ dọn dẹp ổ cứng tự động định kỳ...")
            try:
                cleanup_old_sessions(captures_dir, days_to_keep)
            except Exception as e:
                print(f"[CLEANUP DAEMON ERROR] Lỗi dọn dẹp định kỳ: {e}")

    # Đặt daemon=True để thread tự tắt khi app chính tắt
    t = threading.Thread(target=daemon_job, daemon=True, name="DiskCleanupDaemonThread")
    t.start()
    print(f"[CLEANUP DAEMON] Đã kích hoạt background thread thành công! (Giữ lại: {days_to_keep} ngày, Chu kỳ quét: {interval_hours} giờ)")
    return t
