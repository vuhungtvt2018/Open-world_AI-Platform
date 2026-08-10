import io
from datetime import datetime

from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from services.database.session import SessionLocal

router = APIRouter()

@router.get("/api/reports/lookup/{item_code}")
async def report_lookup(item_code: str):
    db = SessionLocal()
    try:
        from sqlalchemy import text
        insp_count = db.execute(text(f"SELECT COUNT(*) FROM inspection_records WHERE \"ItemCode\" = '{item_code}'")).scalar()
        qc_count = db.execute(text(f"SELECT COUNT(*) FROM qc_product_photo_library WHERE \"ItemCode\" = '{item_code}'")).scalar()
        
        reports = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        if insp_count and insp_count > 0:
            reports.append({
                "id": f"REP-INSP-{item_code}",
                "name": f"Inspection Records - {item_code}",
                "type": "System",
                "format": "Excel",
                "date": now_str,
                "size": f"{insp_count} rows",
                "downloads": 0,
                "lastDownloaded": "-"
            })
            
        if qc_count and qc_count > 0:
            reports.append({
                "id": f"REP-QC-{item_code}",
                "name": f"QC Photo Library - {item_code}",
                "type": "System",
                "format": "Excel",
                "date": now_str,
                "size": f"{qc_count} rows",
                "downloads": 0,
                "lastDownloaded": "-"
            })
            
        return {"status": "success", "reports": reports}
    finally:
        db.close()

@router.get("/api/reports/download/{table_name}/{item_code}")
async def report_download(table_name: str, item_code: str):
    db = SessionLocal()
    try:
        if table_name not in ["inspection_records", "qc_product_photo_library"]:
            return Response(status_code=400, content="Invalid table")
            
        try:
            import pandas as pd
            df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE \"ItemCode\" = '{item_code}'", db.bind)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name=item_code)
                
            output.seek(0)
            headers = {
                'Content-Disposition': f'attachment; filename="{table_name}_{item_code}.xlsx"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
            return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        except ImportError:
            # Fallback to CSV if pandas/openpyxl is not installed
            import csv
            from sqlalchemy import text
            rows = db.execute(text(f"SELECT * FROM {table_name} WHERE \"ItemCode\" = '{item_code}'")).mappings().all()
            if not rows:
                return Response(status_code=404, content="No data found")
            
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
                
            headers = {
                'Content-Disposition': f'attachment; filename="{table_name}_{item_code}.csv"',
                'Access-Control-Expose-Headers': 'Content-Disposition'
            }
            return Response(content=output.getvalue(), media_type="text/csv", headers=headers)
            
    except Exception as e:
        print(f"Error generating report: {e}")
        return Response(status_code=500, content=f"Error generating report: {e}")
    finally:
        db.close()


