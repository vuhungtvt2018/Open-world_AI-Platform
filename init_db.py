from pathlib import Path
import sqlite3


DATABASE_DIR = Path(__file__).resolve().parent


EDGE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS edge_devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_code TEXT NOT NULL UNIQUE,
    device_name TEXT NOT NULL,
    ip_address TEXT,
    mac_address TEXT,

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT
);


CREATE TABLE IF NOT EXISTS qc_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_code TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    image_name TEXT,
    description TEXT,

    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1)),

    version INTEGER NOT NULL DEFAULT 1,

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT
);


CREATE TABLE IF NOT EXISTS detection_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    edge_device_id INTEGER NOT NULL,

    file_name TEXT NOT NULL,
    local_file_path TEXT NOT NULL,

    product_code TEXT,
    product_name TEXT,

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT,

    FOREIGN KEY (edge_device_id)
        REFERENCES edge_devices(id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);


CREATE TABLE IF NOT EXISTS detection_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    detection_image_id INTEGER NOT NULL,
    qc_product_id INTEGER NOT NULL,

    item_qty INTEGER NOT NULL DEFAULT 0
        CHECK (item_qty >= 0),

    bbox TEXT,

    score REAL
        CHECK (
            score IS NULL
            OR (score >= 0 AND score <= 1)
        ),

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT,

    FOREIGN KEY (detection_image_id)
        REFERENCES detection_images(id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    FOREIGN KEY (qc_product_id)
        REFERENCES qc_products(id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);


CREATE INDEX IF NOT EXISTS idx_edge_images_device
ON detection_images(edge_device_id);

CREATE INDEX IF NOT EXISTS idx_edge_results_image
ON detection_results(detection_image_id);

CREATE INDEX IF NOT EXISTS idx_edge_results_product
ON detection_results(qc_product_id);
"""


SERVER_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    site_code TEXT NOT NULL UNIQUE,
    site_name TEXT NOT NULL,
    description TEXT,

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT
);


CREATE TABLE IF NOT EXISTS edge_devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    site_id INTEGER NOT NULL,

    device_code TEXT NOT NULL UNIQUE,
    device_name TEXT NOT NULL,
    ip_address TEXT,
    mac_address TEXT,

    status TEXT NOT NULL DEFAULT 'active',

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT,

    FOREIGN KEY (site_id)
        REFERENCES sites(id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);


CREATE TABLE IF NOT EXISTS qc_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    product_code TEXT NOT NULL UNIQUE,
    product_name TEXT NOT NULL,
    image_name TEXT,
    description TEXT,

    is_active INTEGER NOT NULL DEFAULT 1
        CHECK (is_active IN (0, 1)),

    version INTEGER NOT NULL DEFAULT 1,

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT
);


CREATE TABLE IF NOT EXISTS detection_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    edge_device_id INTEGER NOT NULL,

    file_name TEXT NOT NULL,
    local_file_path TEXT NOT NULL,

    product_code TEXT,
    product_name TEXT,

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT,

    FOREIGN KEY (edge_device_id)
        REFERENCES edge_devices(id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);


CREATE TABLE IF NOT EXISTS detection_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    detection_image_id INTEGER NOT NULL,
    qc_product_id INTEGER NOT NULL,

    item_qty INTEGER NOT NULL DEFAULT 0
        CHECK (item_qty >= 0),

    bbox TEXT,

    score REAL
        CHECK (
            score IS NULL
            OR (score >= 0 AND score <= 1)
        ),

    create_time TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    create_user TEXT,
    update_time TEXT,
    update_user TEXT,

    FOREIGN KEY (detection_image_id)
        REFERENCES detection_images(id)
        ON UPDATE CASCADE
        ON DELETE CASCADE,

    FOREIGN KEY (qc_product_id)
        REFERENCES qc_products(id)
        ON UPDATE CASCADE
        ON DELETE RESTRICT
);


CREATE INDEX IF NOT EXISTS idx_server_devices_site
ON edge_devices(site_id);

CREATE INDEX IF NOT EXISTS idx_server_images_device
ON detection_images(edge_device_id);

CREATE INDEX IF NOT EXISTS idx_server_results_image
ON detection_results(detection_image_id);

CREATE INDEX IF NOT EXISTS idx_server_results_product
ON detection_results(qc_product_id);
"""


def create_database(
    database_name: str,
    schema: str,
) -> None:
    database_path = DATABASE_DIR / database_name

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript(schema)

        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check;"
        ).fetchall()

        if foreign_key_errors:
            raise RuntimeError(
                f"Lỗi khóa ngoại trong {database_name}: "
                f"{foreign_key_errors}"
            )

        connection.commit()

    print(f"Đã tạo thành công: {database_path}")


def main() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    create_database(
        database_name="edge.db",
        schema=EDGE_SCHEMA,
    )

    create_database(
        database_name="server.db",
        schema=SERVER_SCHEMA,
    )

    print("Hoàn tất tạo Edge DB và Server DB.")


if __name__ == "__main__":
    main()