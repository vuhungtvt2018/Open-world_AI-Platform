import time
import requests
import os
import ast
import sqlite3
import yaml
from datetime import datetime
import json

# Đọc cấu hình hoàn toàn độc lập
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

cfg = load_config()
import base64

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), cfg["DB_PATH"]))
EDGE_NODE_ID = cfg.get("EDGE_NODE_ID", "EDGE_001")
SYNC_DEFECT_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../1_edge_node/sync_defect_image"))
os.makedirs(SYNC_DEFECT_IMAGE_DIR, exist_ok=True)

def get_last_sync_time(cursor, key="last_inspection_sync_time"):
    cursor.execute("SELECT value FROM sync_states WHERE key = ?", (key,))
    row = cursor.fetchone()
    return row[0] if row else None

def set_last_sync_time(conn, cursor, value, key="last_inspection_sync_time"):
    cursor.execute("SELECT value FROM sync_states WHERE key = ?", (key,))
    if cursor.fetchone():
        cursor.execute("UPDATE sync_states SET value = ?, updated_at = CURRENT_TIMESTAMP WHERE key = ?", (value, key))
    else:
        cursor.execute("INSERT INTO sync_states (key, value) VALUES (?, ?)", (key, value))
    conn.commit()

def sync_job():
    print(f"[SYNC DAEMON] Bắt đầu tiến trình độc lập. Quét CSDL tại: {DB_PATH}")
    while True:
        if not os.path.exists(DB_PATH):
            time.sleep(2)
            continue
            
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Đảm bảo bảng sync_states tồn tại
            cursor.execute('''CREATE TABLE IF NOT EXISTS sync_states (
                                key VARCHAR(100) PRIMARY KEY,
                                value VARCHAR(255),
                                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                              )''')
            
            # Đảm bảo bảng qc_product_photo_library tồn tại
            cursor.execute('''CREATE TABLE IF NOT EXISTS qc_product_photo_library (
                                ID INTEGER PRIMARY KEY,
                                SM_ID VARCHAR(50) UNIQUE,
                                ItemCode VARCHAR(100),
                                ImageName VARCHAR(255),
                                ImageType INTEGER,
                                ErrorDetail VARCHAR(255),
                                Insert_PIC VARCHAR(100),
                                Insert_Date DATETIME,
                                Update_PIC VARCHAR(100),
                                Update_Date DATETIME
                              )''')
            conn.commit()
            
            last_sync = get_last_sync_time(cursor)
            
            # Fetch records
            if last_sync:
                cursor.execute("SELECT id, timestamp, original_image, ng_detected, latency_ms, created_at FROM inspection_records WHERE created_at > ? ORDER BY created_at ASC LIMIT 50", (last_sync,))
            else:
                cursor.execute("SELECT id, timestamp, original_image, ng_detected, latency_ms, created_at FROM inspection_records ORDER BY created_at ASC LIMIT 50")
            
            records = cursor.fetchall()
            
            for record in records:
                rec_id, rec_ts, rec_img, rec_ng_detected, rec_latency, rec_created = record
                
                # Fetch objects for this record
                cursor.execute("SELECT id, object_index, bbox, score, is_ng, overlap_ratio, crop_image FROM bolt_objects WHERE record_id = ?", (rec_id,))
                objects = cursor.fetchall()
                
                object_details = []
                for obj in objects:
                    obj_id, object_index, bbox, score, is_ng, overlap_ratio, crop_image = obj
                    
                    # Fetch anomalies
                    cursor.execute("SELECT id, bbox_full, bbox_in_object_crop, defect_class, similarity FROM anomaly_details WHERE bolt_id = ?", (obj_id,))
                    anomalies = cursor.fetchall()
                    
                    anomaly_details = []
                    for anomaly in anomalies:
                        ano_id, bbox_full, bbox_in_object_crop, defect_class, similarity = anomaly
                        anomaly_details.append({
                            "id": ano_id,
                            "bbox_full": bbox_full,
                            "bbox_in_object_crop": bbox_in_object_crop,
                            "defect_class": defect_class,
                            "similarity": similarity
                        })
                    
                    object_details.append({
                        "id": obj_id,
                        "object_index": object_index,
                        "bbox": bbox,
                        "score": score,
                        "is_ng": is_ng,
                        "overlap_ratio": overlap_ratio,
                        "crop_image": crop_image,
                        "anomalies": anomaly_details
                    })
                    
                payload = {
                    "edge_node_id": EDGE_NODE_ID,
                    "record_id": rec_id,
                    "timestamp": rec_ts,
                    "ng_detected": bool(rec_ng_detected),
                    "latency_ms": float(rec_latency),
                    "created_at": rec_created,
                    "item_code": cfg.get("PRODUCT_NAME", "unknown"),
                    "objects": object_details
                }
                
                if cfg.get("API_SYNC_ENABLE") and cfg.get("API_SYNC_UP_URL"):
                    try:
                        # Đính kèm hình ảnh
                        files = None
                        if rec_img and os.path.exists(rec_img):
                            # Mở file ở chế độ đọc binary
                            files = {'file': (os.path.basename(rec_img), open(rec_img, 'rb'), 'image/jpeg')}
                            
                        print(f"[SYNC WORKER] Pushing record {rec_id} to Server...")
                        r = requests.post(
                            cfg["API_SYNC_UP_URL"], 
                            data={"payload": json.dumps(payload)}, 
                            files=files,
                            timeout=10, 
                            proxies={"http": None, "https": None}
                        )
                        if files:
                            files['file'][1].close() # Đóng file
                            
                        if not (200 <= r.status_code < 300):
                            print(f"[SYNC WORKER] HTTP {r.status_code}: {r.text}")
                        else:
                            print(f"[SYNC WORKER] Thang cong dong bo record {rec_id}")
                    except Exception as e:
                        print(f"[SYNC WORKER] API Error: {e}")
                        if 'files' in locals() and files and not files['file'][1].closed:
                            files['file'][1].close()
                        raise e # Dừng để retry sau
                        
                # Cập nhật thời gian thành công
                set_last_sync_time(conn, cursor, rec_created)
                
            # Đồng bộ hướng xuống (Server -> Edge) (Cấu hình)
            if cfg.get("API_SYNC_ENABLE") and cfg.get("API_SYNC_DOWN_URL"):
                try:
                    r_down = requests.get(cfg["API_SYNC_DOWN_URL"], timeout=5, proxies={"http": None, "https": None})
                    if r_down.status_code == 200:
                        data_down = r_down.json()
                        if data_down.get("command") == "update_config":
                            edge_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../1_edge_node/config.yaml"))
                            with open(edge_cfg_path, "r", encoding="utf-8") as f:
                                edge_cfg = yaml.safe_load(f)
                            
                            updated = False
                            for k, v in data_down.get("data", {}).items():
                                if k in edge_cfg and edge_cfg[k] != v:
                                    edge_cfg[k] = v
                                    updated = True
                            
                            if updated:
                                with open(edge_cfg_path, "w", encoding="utf-8") as f:
                                    yaml.dump(edge_cfg, f, default_flow_style=False, sort_keys=False)
                                print("[SYNC DOWN] Da ghi de file config.yaml thanh cong!")
                except Exception as e:
                    pass
                    
            # Đồng bộ Thư viện mẫu (Server -> Edge)
            if cfg.get("API_SYNC_ENABLE") and cfg.get("API_SYNC_LIBRARY_DOWN_URL"):
                try:
                    last_lib_sync = get_last_sync_time(cursor, "last_library_sync_time")
                    url = cfg["API_SYNC_LIBRARY_DOWN_URL"]
                    if last_lib_sync:
                        url += f"?last_update={last_lib_sync}"
                        
                    r_lib = requests.get(url, timeout=5, proxies={"http": None, "https": None})
                    if r_lib.status_code == 200:
                        res_data = r_lib.json()
                        records = res_data.get("data", [])
                        
                        max_update = last_lib_sync or "1970-01-01"
                        
                        for rec in records:
                            # Save physical image if base64 data is present
                            img_b64 = rec.get("image_base64")
                            img_name = rec.get("ImageName")
                            item_code = rec.get("ItemCode")
                            if img_b64 and img_name:
                                try:
                                    img_data = base64.b64decode(img_b64)
                                    save_dir = os.path.join(SYNC_DEFECT_IMAGE_DIR, item_code) if item_code else SYNC_DEFECT_IMAGE_DIR
                                    os.makedirs(save_dir, exist_ok=True)
                                    img_path = os.path.join(save_dir, img_name)
                                    with open(img_path, "wb") as f:
                                        f.write(img_data)
                                except Exception as e:
                                    print(f"[SYNC DOWN] Lỗi khi lưu ảnh {img_name}: {e}")

                            # Upsert vao sqlite
                            cursor.execute('''
                                INSERT INTO qc_product_photo_library (
                                    ID, SM_ID, ItemCode, ImageName, ImageType, 
                                    ErrorDetail, Insert_PIC, Insert_Date, Update_PIC, Update_Date
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ON CONFLICT(ID) DO UPDATE SET
                                    SM_ID=excluded.SM_ID,
                                    ItemCode=excluded.ItemCode,
                                    ImageName=excluded.ImageName,
                                    ImageType=excluded.ImageType,
                                    ErrorDetail=excluded.ErrorDetail,
                                    Insert_PIC=excluded.Insert_PIC,
                                    Insert_Date=excluded.Insert_Date,
                                    Update_PIC=excluded.Update_PIC,
                                    Update_Date=excluded.Update_Date
                            ''', (
                                rec.get("ID"), rec.get("SM_ID"), rec.get("ItemCode"), 
                                rec.get("ImageName"), rec.get("ImageType"), rec.get("ErrorDetail"),
                                rec.get("Insert_PIC"), rec.get("Insert_Date"), 
                                rec.get("Update_PIC"), rec.get("Update_Date")
                            ))
                            
                            u_date = rec.get("Update_Date")
                            if u_date and u_date > max_update:
                                max_update = u_date
                                
                        if records:
                            set_last_sync_time(conn, cursor, max_update, "last_library_sync_time")
                            print(f"[SYNC DOWN] Da cap nhat {len(records)} ban ghi thu vien mau.")
                except Exception as e:
                    print(f"[SYNC DOWN] Loi dong bo thu vien: {e}")
                
        except Exception as e:
            # Lỗi mạng hoặc DB bị lock, thử lại sau
            pass
        finally:
            if 'conn' in locals():
                conn.close()
                
        time.sleep(2) # Quét mỗi 2 giây

if __name__ == "__main__":
    sync_job()
