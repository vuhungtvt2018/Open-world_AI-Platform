from sqlalchemy.orm import Session

from .models import Product, ProductType
from .session import SessionLocal


def initialize_default_product() -> None:
    db: Session = SessionLocal()

    try:
        product_type = (
            db.query(ProductType)
            .filter(ProductType.code == "BOLT")
            .first()
        )

        if product_type is None:
            product_type = ProductType(
                code="BOLT",
                name="Bolt",
                description="Bolt product type",
            )
            db.add(product_type)
            db.flush()

        product = (
            db.query(Product)
            .filter(Product.item_code == "BOLT-001")
            .first()
        )

        if product is None:
            product = Product(
                item_code="BOLT-001",
                name="Bolt 001",
                product_type_id=product_type.id,
                description="Default bolt product",
            )
            db.add(product)

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()