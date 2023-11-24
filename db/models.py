from sqlalchemy.types import UserDefinedType
from sqlalchemy import DateTime, Column, Integer, func, VARCHAR, DOUBLE, text
from sqlalchemy.orm import declarative_base

from db.connection import SQLServerConnection as ssc

import pandas as pd


class Geometry(UserDefinedType):
    """
    MSSQL Geometry datatype
    """

    cache_ok = True

    def get_col_spec(self):
        return "GEOMETRY"

    def bind_expression(self, bindvalue):
        expression = text(
            f"GEOMETRY::STGeomFromText(:{bindvalue.key}, 4326)"
        ).bindparams(bindvalue)

        return expression


# Base class
Base = declarative_base()


class ImagePredictions(Base):
    __tablename__ = "IMAGE_PREDICTIONS_SNOW"
    __table_args__ = {"schema": "BO_DATA"}

    OBJECTID = Column(Integer, primary_key=True, autoincrement=False)
    Shape = Column(Geometry)
    image_name = Column(VARCHAR)
    datetime_processed = Column(DateTime)
    prediction_prob = Column(DOUBLE)
    image_link = Column(VARCHAR)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now())

    @classmethod
    def read_all(cls):
        results = pd.read_sql(
            f"SELECT * FROM {cls.__table_args__['schema']}.{cls.__tablename__}",
            ssc().engine(),
        )
        return results
